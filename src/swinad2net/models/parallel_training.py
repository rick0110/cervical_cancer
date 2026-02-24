"""Parallel Training Manager for Multi-Core Training.

This module provides utilities for training multiple models simultaneously
across CPU cores, with support for:
- Process-based parallelism (multiprocessing)
- Shared memory for result aggregation
- Dynamic load balancing
- Fault tolerance for failed workers
- Real-time comparison of results
"""

import os
import torch
import torch.multiprocessing as mp
from multiprocessing import Manager, Queue, Process, cpu_count
from typing import Dict, List, Tuple, Callable, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import time
import traceback
import pickle
import json
import numpy as np
from queue import Empty


@dataclass
class TrainingJob:
    """Represents a training job to be executed."""
    job_id: int
    config: Dict[str, Any]
    epochs: int
    trial_id: Optional[int] = None
    resume_from: Optional[str] = None  # Path to checkpoint to resume from
    
    
@dataclass 
class TrainingResult:
    """Result from a completed training job."""
    job_id: int
    trial_id: Optional[int]
    config: Dict[str, Any]
    metrics: Dict[str, float]
    history: Dict[str, List[float]]
    checkpoint_path: str
    status: str  # 'completed', 'failed', 'stopped'
    error_message: Optional[str] = None
    training_time: float = 0.0


class SharedResultsManager:
    """
    Manager for shared results across processes.
    Enables real-time comparison of training progress.
    """
    
    def __init__(self, manager: Any):
        """
        Args:
            manager: multiprocessing.Manager instance
        """
        self._lock = manager.Lock()
        self._results = manager.dict()
        self._best_metrics = manager.dict()
        self._intermediate = manager.dict()  # job_id -> list of (epoch, metrics)
        
    def report_intermediate(self, job_id: int, epoch: int, metrics: Dict[str, float]):
        """Report intermediate results from a running job."""
        with self._lock:
            key = str(job_id)
            if key not in self._intermediate:
                self._intermediate[key] = []
            current = list(self._intermediate.get(key, []))
            current.append((epoch, dict(metrics)))
            self._intermediate[key] = current
            
    def report_final(self, job_id: int, result: TrainingResult):
        """Report final results from a completed job."""
        with self._lock:
            self._results[str(job_id)] = result
            
    def get_best_metric(self, metric_name: str, mode: str = 'min') -> Optional[float]:
        """Get the best value for a metric across all jobs."""
        with self._lock:
            best = None
            for key in self._intermediate.keys():
                history = self._intermediate.get(key, [])
                for epoch, metrics in history:
                    if metric_name in metrics:
                        val = metrics[metric_name]
                        if best is None:
                            best = val
                        elif mode == 'min' and val < best:
                            best = val
                        elif mode == 'max' and val > best:
                            best = val
            return best
            
    def get_all_results(self) -> Dict[int, TrainingResult]:
        """Get all completed results."""
        with self._lock:
            return {int(k): v for k, v in self._results.items()}
            
    def get_intermediate(self, job_id: int) -> List[Tuple[int, Dict]]:
        """Get intermediate results for a specific job."""
        with self._lock:
            return list(self._intermediate.get(str(job_id), []))


def worker_train_model(job_queue: Queue,
                       result_queue: Queue,
                       shared_results: SharedResultsManager,
                       train_fn: Callable,
                       config_static: Dict[str, Any],
                       device: str,
                       worker_id: int):
    """
    Worker function for training models.
    Runs in a separate process and pulls jobs from the queue.
    
    Args:
        job_queue: Queue of TrainingJob objects
        result_queue: Queue to put TrainingResult objects
        shared_results: SharedResultsManager for intermediate results
        train_fn: Training function to call
        config_static: Static configuration (device, data paths, etc.)
        device: Device for this worker
        worker_id: Worker ID for logging
    """
    print(f"Worker {worker_id} started on device {device}")
    
    while True:
        try:
            # Get job with timeout
            job = job_queue.get(timeout=5)
            
            if job is None:  # Poison pill
                print(f"Worker {worker_id} received shutdown signal")
                break
                
            print(f"Worker {worker_id}: Starting job {job.job_id}")
            start_time = time.time()
            
            try:
                # Combine static config with job config
                full_config = {**config_static, **job.config}
                full_config['epochs'] = job.epochs
                full_config['device'] = device
                full_config['job_id'] = job.job_id
                full_config['checkpoint_path'] = job.resume_from
                
                # Create callback for intermediate results
                def intermediate_callback(epoch: int, metrics: Dict[str, float]):
                    shared_results.report_intermediate(job.job_id, epoch, metrics)
                    # Return best metric for adaptive early stopping comparison
                    return shared_results.get_best_metric('val_loss', 'min')
                    
                full_config['intermediate_callback'] = intermediate_callback
                
                # Run training
                result = train_fn(**full_config)
                
                training_time = time.time() - start_time
                
                training_result = TrainingResult(
                    job_id=job.job_id,
                    trial_id=job.trial_id,
                    config=job.config,
                    metrics=result.get('metrics', {}),
                    history=result.get('history', {}),
                    checkpoint_path=result.get('checkpoint_path', ''),
                    status='completed',
                    training_time=training_time
                )
                
            except Exception as e:
                traceback.print_exc()
                training_time = time.time() - start_time
                training_result = TrainingResult(
                    job_id=job.job_id,
                    trial_id=job.trial_id,
                    config=job.config,
                    metrics={},
                    history={},
                    checkpoint_path='',
                    status='failed',
                    error_message=str(e),
                    training_time=training_time
                )
                
            shared_results.report_final(job.job_id, training_result)
            result_queue.put(training_result)
            print(f"Worker {worker_id}: Completed job {job.job_id} in {training_time:.1f}s")
            
        except Empty:
            # No job available, check if we should exit
            continue
        except Exception as e:
            print(f"Worker {worker_id} error: {e}")
            traceback.print_exc()
            

class ParallelTrainingManager:
    """
    Manager for parallel model training across multiple CPU cores.
    """
    
    def __init__(self,
                 num_workers: Optional[int] = None,
                 devices: Optional[List[str]] = None,
                 checkpoint_base_dir: str = "parallel_checkpoints"):
        """
        Args:
            num_workers: Number of parallel workers (default: cpu_count())
            devices: List of devices for workers (default: auto-detect)
            checkpoint_base_dir: Base directory for checkpoints
        """
        self.num_workers = num_workers or max(1, cpu_count() - 1)
        self.checkpoint_base_dir = checkpoint_base_dir
        os.makedirs(checkpoint_base_dir, exist_ok=True)
        
        # Auto-detect devices
        if devices is None:
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                # Distribute workers across GPUs
                self.devices = [f"cuda:{i % gpu_count}" for i in range(self.num_workers)]
            else:
                self.devices = ["cpu"] * self.num_workers
        else:
            self.devices = devices
            
        self.workers: List[Process] = []
        self.job_queue: Optional[Queue] = None
        self.result_queue: Optional[Queue] = None
        self.shared_results: Optional[SharedResultsManager] = None
        self.manager: Optional[Any] = None
        
        self.job_counter = 0
        self.pending_jobs: Dict[int, TrainingJob] = {}
        self.completed_results: List[TrainingResult] = []
        
        print(f"\n{'='*60}")
        print(f"Parallel Training Manager")
        print(f"Workers: {self.num_workers}")
        print(f"Devices: {set(self.devices)}")
        print(f"{'='*60}\n")
        
    def start_workers(self, train_fn: Callable, config_static: Dict[str, Any]):
        """
        Start worker processes.
        
        Args:
            train_fn: Training function to use
            config_static: Static configuration for all jobs
        """
        # Set multiprocessing start method
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # Already set
            
        self.manager = Manager()
        self.job_queue = Queue()
        self.result_queue = Queue()
        self.shared_results = SharedResultsManager(self.manager)
        
        for i in range(self.num_workers):
            p = Process(
                target=worker_train_model,
                args=(
                    self.job_queue,
                    self.result_queue,
                    self.shared_results,
                    train_fn,
                    config_static,
                    self.devices[i],
                    i
                )
            )
            p.start()
            self.workers.append(p)
            
        print(f"Started {self.num_workers} worker processes")
        
    def submit_job(self, config: Dict[str, Any], epochs: int,
                   trial_id: Optional[int] = None,
                   resume_from: Optional[str] = None) -> int:
        """
        Submit a training job.
        
        Args:
            config: Hyperparameter configuration
            epochs: Number of epochs to train
            trial_id: Optional trial ID for Hyperband integration
            resume_from: Optional checkpoint path to resume from
            
        Returns:
            Job ID
        """
        job = TrainingJob(
            job_id=self.job_counter,
            config=config,
            epochs=epochs,
            trial_id=trial_id,
            resume_from=resume_from
        )
        self.pending_jobs[job.job_id] = job
        self.job_queue.put(job)
        self.job_counter += 1
        return job.job_id
        
    def submit_jobs_batch(self, configs: List[Dict[str, Any]], epochs: int) -> List[int]:
        """Submit multiple jobs at once."""
        return [self.submit_job(cfg, epochs) for cfg in configs]
        
    def collect_results(self, timeout: float = None) -> List[TrainingResult]:
        """
        Collect completed results.
        
        Args:
            timeout: Optional timeout in seconds
            
        Returns:
            List of TrainingResult objects
        """
        results = []
        start_time = time.time()
        
        while True:
            try:
                result = self.result_queue.get(timeout=1.0)
                results.append(result)
                self.completed_results.append(result)
                
                if result.job_id in self.pending_jobs:
                    del self.pending_jobs[result.job_id]
                    
            except Empty:
                pass
                
            if timeout and (time.time() - start_time) > timeout:
                break
            if not self.pending_jobs:
                break
                
        return results
        
    def wait_for_completion(self) -> List[TrainingResult]:
        """Wait for all pending jobs to complete."""
        while self.pending_jobs:
            self.collect_results(timeout=5)
        return self.completed_results
        
    def stop_workers(self):
        """Stop all worker processes."""
        # Send poison pills
        for _ in self.workers:
            self.job_queue.put(None)
            
        # Wait for workers to finish
        for p in self.workers:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
                
        self.workers = []
        if self.manager:
            self.manager.shutdown()
            
        print("All workers stopped")
        
    def get_best_result(self, metric: str = 'val_accuracy', mode: str = 'max') -> TrainingResult:
        """Get the best result based on a metric."""
        if not self.completed_results:
            raise ValueError("No completed results")
            
        if mode == 'max':
            return max(self.completed_results, 
                      key=lambda r: r.metrics.get(metric, float('-inf')))
        else:
            return min(self.completed_results,
                      key=lambda r: r.metrics.get(metric, float('inf')))
            
    def get_comparison_report(self) -> str:
        """Generate a comparison report of all completed jobs."""
        if not self.completed_results:
            return "No completed results"
            
        report = f"\n{'='*80}\n"
        report += "PARALLEL TRAINING RESULTS COMPARISON\n"
        report += f"{'='*80}\n\n"
        
        # Sort by val_accuracy descending
        sorted_results = sorted(
            self.completed_results,
            key=lambda r: r.metrics.get('val_accuracy', 0),
            reverse=True
        )
        
        for i, result in enumerate(sorted_results, 1):
            report += f"Rank {i}: Job {result.job_id}\n"
            report += f"  Status: {result.status}\n"
            report += f"  Time: {result.training_time:.1f}s\n"
            report += f"  Metrics:\n"
            for k, v in result.metrics.items():
                report += f"    {k}: {v:.4f}\n"
            report += f"  Config:\n"
            for k, v in result.config.items():
                report += f"    {k}: {v}\n"
            report += "\n"
            
        return report
        
    def save_results(self, path: str):
        """Save all results to disk."""
        data = {
            'completed_results': self.completed_results,
            'timestamp': datetime.now().isoformat()
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Results saved to {path}")


class ResultComparator:
    """
    Real-time comparator for training results across parallel jobs.
    """
    
    def __init__(self, shared_results: SharedResultsManager,
                 metrics_to_track: List[str] = None):
        """
        Args:
            shared_results: SharedResultsManager instance
            metrics_to_track: List of metric names to track
        """
        self.shared_results = shared_results
        self.metrics = metrics_to_track or ['val_loss', 'val_accuracy']
        
    def get_leaderboard(self, metric: str = 'val_accuracy', 
                        mode: str = 'max') -> List[Dict]:
        """
        Get current leaderboard based on a metric.
        
        Returns:
            List of dicts with job_id and metric value, sorted by performance
        """
        leaderboard = []
        results = self.shared_results.get_all_results()
        
        for job_id, result in results.items():
            if metric in result.metrics:
                leaderboard.append({
                    'job_id': job_id,
                    metric: result.metrics[metric],
                    'config': result.config
                })
                
        # Sort
        reverse = (mode == 'max')
        leaderboard.sort(key=lambda x: x[metric], reverse=reverse)
        
        return leaderboard
        
    def get_convergence_comparison(self) -> Dict[int, List[float]]:
        """
        Get convergence curves for all jobs.
        
        Returns:
            Dict mapping job_id to list of val_loss values
        """
        convergence = {}
        
        for job_id in range(100):  # Arbitrary max
            history = self.shared_results.get_intermediate(job_id)
            if history:
                convergence[job_id] = [m.get('val_loss', np.nan) 
                                       for _, m in history]
                                       
        return convergence
        
    def identify_poor_performers(self, 
                                  metric: str = 'val_loss',
                                  threshold_percentile: float = 75) -> List[int]:
        """
        Identify jobs performing worse than the threshold percentile.
        
        Returns:
            List of job_ids that are poor performers
        """
        all_metrics = []
        job_metrics = {}
        
        for job_id in range(100):
            history = self.shared_results.get_intermediate(job_id)
            if history and len(history) >= 5:  # Need some epochs
                latest = history[-1][1].get(metric)
                if latest is not None:
                    all_metrics.append(latest)
                    job_metrics[job_id] = latest
                    
        if len(all_metrics) < 2:
            return []
            
        threshold = np.percentile(all_metrics, threshold_percentile)
        
        # For loss, high values are bad
        poor = [jid for jid, val in job_metrics.items() if val > threshold]
        
        return poor
