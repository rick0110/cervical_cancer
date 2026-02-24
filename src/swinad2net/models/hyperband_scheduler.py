"""Hyperband Scheduler for Hyperparameter Optimization.

This module implements the Hyperband algorithm for efficient hyperparameter
optimization. Hyperband uses early stopping to allocate resources efficiently
by eliminating poor configurations early.

Key concepts:
- Successive Halving: Train multiple configurations, periodically eliminate the worst
- Hyperband: Run multiple rounds of Successive Halving with different initial budgets

Reference:
    Li, L., et al. "Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization"
    https://arxiv.org/abs/1603.06560
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Callable, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import json
import os
import pickle
from datetime import datetime
import random


class TrialStatus(Enum):
    """Status of a hyperparameter trial."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    PRUNED = "pruned"
    FAILED = "failed"


@dataclass
class Trial:
    """Represents a single hyperparameter configuration trial."""
    trial_id: int
    config: Dict[str, Any]
    bracket: int
    rung: int
    budget: int  # epochs allocated
    status: TrialStatus = TrialStatus.PENDING
    metrics: Dict[str, float] = field(default_factory=dict)
    intermediate_metrics: List[Dict[str, float]] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    state_dict_path: Optional[str] = None
    
    def record_intermediate(self, epoch: int, metrics: Dict[str, float]):
        """Record metrics at an intermediate checkpoint."""
        self.intermediate_metrics.append({
            'epoch': epoch,
            **metrics
        })
        
    def mark_completed(self, final_metrics: Dict[str, float]):
        """Mark trial as completed with final metrics."""
        self.status = TrialStatus.COMPLETED
        self.metrics = final_metrics
        self.end_time = datetime.now()
        
    def mark_failed(self, error: str):
        """Mark trial as failed with error message."""
        self.status = TrialStatus.FAILED
        self.error_message = error
        self.end_time = datetime.now()
        
    def mark_pruned(self, reason: str = ""):
        """Mark trial as pruned (early stopped)."""
        self.status = TrialStatus.PRUNED
        self.error_message = reason
        self.end_time = datetime.now()


@dataclass
class HyperbandBracket:
    """A single bracket in Hyperband containing multiple rungs."""
    bracket_id: int
    n_configs: int  # Initial number of configurations
    min_budget: int  # Minimum budget per configuration (epochs)
    max_budget: int
    eta: int  # Reduction factor
    rungs: List[Tuple[int, int]] = field(default_factory=list)  # (n_configs, budget) per rung
    trials: List[Trial] = field(default_factory=list)
    current_rung: int = 0
    
    def __post_init__(self):
        """Calculate rungs after initialization."""
        if not self.rungs:
            self._compute_rungs()
            
    def _compute_rungs(self):
        """Compute the number of configs and budget for each rung."""
        s = self.bracket_id
        n = self.n_configs
        r = self.min_budget
        
        self.rungs = []
        for i in range(s + 1):
            n_i = int(math.floor(n * (self.eta ** (-i))))
            r_i = int(r * (self.eta ** i))
            self.rungs.append((max(1, n_i), min(r_i, self.max_budget)))


class HyperbandScheduler:
    """
    Hyperband scheduler for hyperparameter optimization.
    
    Hyperband explores the space of hyperparameters by running multiple
    "brackets" of trials. Each bracket starts with different initial
    configurations and budgets, implementing successive halving.
    """
    
    def __init__(self,
                 max_budget: int = 100,
                 eta: int = 3,
                 metric: str = "val_loss",
                 mode: str = "min",
                 checkpoint_dir: str = "hyperband_checkpoints",
                 seed: int = 42):
        """
        Initialize Hyperband scheduler.
        
        Args:
            max_budget: Maximum budget (epochs) for any single configuration
            eta: Reduction factor (typically 3)
            metric: Metric to optimize
            mode: "min" or "max" for the metric
            checkpoint_dir: Directory to save trial checkpoints
            seed: Random seed for reproducibility
        """
        self.max_budget = max_budget
        self.eta = eta
        self.metric = metric
        self.mode = mode
        self.checkpoint_dir = checkpoint_dir
        self.seed = seed
        
        # Calculate number of brackets
        self.s_max = int(math.floor(math.log(max_budget) / math.log(eta)))
        self.B = (self.s_max + 1) * max_budget  # Total budget per bracket
        
        self.brackets: List[HyperbandBracket] = []
        self.all_trials: List[Trial] = []
        self.trial_counter = 0
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        random.seed(seed)
        np.random.seed(seed)
        
        print(f"\n{'='*60}")
        print(f"Hyperband Scheduler Initialized")
        print(f"Max Budget: {max_budget} | Eta: {eta} | Brackets: {self.s_max + 1}")
        print(f"Optimizing: {metric} ({mode})")
        print(f"{'='*60}\n")
        
    def _init_brackets(self, config_space: Dict[str, List[Any]]):
        """Initialize all brackets with sampled configurations."""
        self.brackets = []
        
        for s in range(self.s_max, -1, -1):
            # Number of initial configs for this bracket
            n = int(math.ceil((self.B / self.max_budget) * (self.eta ** s) / (s + 1)))
            r = self.max_budget * (self.eta ** (-s))  # Min budget
            
            bracket = HyperbandBracket(
                bracket_id=s,
                n_configs=n,
                min_budget=max(1, int(r)),
                max_budget=self.max_budget,
                eta=self.eta
            )
            
            # Sample configurations for this bracket
            for _ in range(n):
                config = self._sample_config(config_space)
                trial = Trial(
                    trial_id=self.trial_counter,
                    config=config,
                    bracket=s,
                    rung=0,
                    budget=bracket.rungs[0][1]
                )
                bracket.trials.append(trial)
                self.all_trials.append(trial)
                self.trial_counter += 1
                
            self.brackets.append(bracket)
            
            print(f"Bracket {s}: {n} configs, min_budget={bracket.rungs[0][1]}, "
                  f"max_budget={bracket.rungs[-1][1]}")
            
    def _sample_config(self, config_space: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Sample a random configuration from the config space."""
        config = {}
        for key, values in config_space.items():
            if isinstance(values, tuple) and len(values) == 3:
                # (min, max, scale) - continuous parameter
                low, high, scale = values
                if scale == 'log':
                    val = np.exp(np.random.uniform(np.log(low), np.log(high)))
                else:
                    val = np.random.uniform(low, high)
                config[key] = val
            elif isinstance(values, tuple) and len(values) == 2:
                # (min, max) - integer parameter
                config[key] = np.random.randint(values[0], values[1] + 1)
            elif isinstance(values, list):
                # Categorical parameter
                config[key] = random.choice(values)
            else:
                config[key] = values
        return config
    
    def _compare_metrics(self, metric1: float, metric2: float) -> bool:
        """Compare metrics. Returns True if metric1 is better than metric2."""
        if self.mode == "min":
            return metric1 < metric2
        return metric1 > metric2
    
    def _get_best_trials(self, trials: List[Trial], n: int) -> List[Trial]:
        """Get the best n trials based on the metric."""
        completed = [t for t in trials if t.status == TrialStatus.COMPLETED]
        
        if not completed:
            return []
            
        # Sort by metric
        reverse = (self.mode == "max")
        sorted_trials = sorted(
            completed,
            key=lambda t: t.metrics.get(self.metric, float('inf') if self.mode == 'min' else float('-inf')),
            reverse=reverse
        )
        
        return sorted_trials[:n]
    
    def get_next_trials(self) -> List[Trial]:
        """
        Get the next batch of trials to run.
        
        Returns:
            List of Trial objects to execute
        """
        pending_trials = []
        
        for bracket in self.brackets:
            # Get pending trials in current rung
            for trial in bracket.trials:
                if trial.status == TrialStatus.PENDING and trial.rung == bracket.current_rung:
                    pending_trials.append(trial)
                    
        return pending_trials
    
    def report_trial_result(self, trial: Trial, metrics: Dict[str, float], 
                            state_dict_path: Optional[str] = None):
        """
        Report the result of a completed trial.
        
        Args:
            trial: The trial that was executed
            metrics: Final metrics from training
            state_dict_path: Path to saved model state dict
        """
        trial.mark_completed(metrics)
        trial.state_dict_path = state_dict_path
        
        # Check if we should advance the bracket to next rung
        bracket = self.brackets[self.s_max - trial.bracket]
        
        # Count completed trials in current rung
        current_rung_trials = [t for t in bracket.trials 
                               if t.rung == bracket.current_rung]
        completed_in_rung = [t for t in current_rung_trials 
                            if t.status == TrialStatus.COMPLETED]
        
        if len(completed_in_rung) == len(current_rung_trials):
            self._advance_bracket(bracket)
            
    def report_trial_failure(self, trial: Trial, error: str):
        """Report a failed trial."""
        trial.mark_failed(error)
        
    def _advance_bracket(self, bracket: HyperbandBracket):
        """Advance a bracket to the next rung after completing current rung."""
        if bracket.current_rung >= len(bracket.rungs) - 1:
            print(f"Bracket {bracket.bracket_id} completed all rungs")
            return
            
        current_rung = bracket.current_rung
        next_rung = current_rung + 1
        
        # Get completed trials from current rung
        current_trials = [t for t in bracket.trials 
                         if t.rung == current_rung and t.status == TrialStatus.COMPLETED]
        
        # Calculate how many to promote
        n_promote = bracket.rungs[next_rung][0]
        next_budget = bracket.rungs[next_rung][1]
        
        # Get best trials
        best_trials = self._get_best_trials(current_trials, n_promote)
        
        # Mark non-promoted trials as pruned
        best_ids = {t.trial_id for t in best_trials}
        for trial in current_trials:
            if trial.trial_id not in best_ids:
                trial.mark_pruned("Not promoted to next rung")
                
        # Create new trials for promoted configs
        for old_trial in best_trials:
            new_trial = Trial(
                trial_id=self.trial_counter,
                config=old_trial.config.copy(),
                bracket=bracket.bracket_id,
                rung=next_rung,
                budget=next_budget,
                state_dict_path=old_trial.state_dict_path  # Resume from checkpoint
            )
            bracket.trials.append(new_trial)
            self.all_trials.append(new_trial)
            self.trial_counter += 1
            
        bracket.current_rung = next_rung
        
        print(f"\nBracket {bracket.bracket_id}: Advanced to rung {next_rung}")
        print(f"Promoted {len(best_trials)} trials with budget {next_budget}")
        
    def is_complete(self) -> bool:
        """Check if all brackets have completed."""
        for bracket in self.brackets:
            if bracket.current_rung < len(bracket.rungs) - 1:
                # Check if there are pending trials
                pending = [t for t in bracket.trials 
                          if t.status == TrialStatus.PENDING]
                if pending:
                    return False
        return True
    
    def get_best_config(self) -> Tuple[Dict[str, Any], float, Trial]:
        """
        Get the best configuration found.
        
        Returns:
            Tuple of (best_config, best_metric, best_trial)
        """
        completed = [t for t in self.all_trials 
                    if t.status == TrialStatus.COMPLETED]
        
        if not completed:
            raise ValueError("No completed trials found")
            
        best = self._get_best_trials(completed, 1)[0]
        return best.config, best.metrics.get(self.metric), best
    
    def save_state(self, path: Optional[str] = None):
        """Save scheduler state to disk."""
        path = path or os.path.join(self.checkpoint_dir, "hyperband_state.pkl")
        state = {
            'max_budget': self.max_budget,
            'eta': self.eta,
            'metric': self.metric,
            'mode': self.mode,
            's_max': self.s_max,
            'trial_counter': self.trial_counter,
            'brackets': self.brackets,
            'all_trials': self.all_trials
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        print(f"Saved Hyperband state to {path}")
        
    def load_state(self, path: Optional[str] = None):
        """Load scheduler state from disk."""
        path = path or os.path.join(self.checkpoint_dir, "hyperband_state.pkl")
        if os.path.exists(path):
            with open(path, 'rb') as f:
                state = pickle.load(f)
            self.max_budget = state['max_budget']
            self.eta = state['eta']
            self.metric = state['metric']
            self.mode = state['mode']
            self.s_max = state['s_max']
            self.trial_counter = state['trial_counter']
            self.brackets = state['brackets']
            self.all_trials = state['all_trials']
            print(f"Loaded Hyperband state from {path}")
            return True
        return False
    
    def get_progress_summary(self) -> str:
        """Get a summary of current progress."""
        total = len(self.all_trials)
        completed = sum(1 for t in self.all_trials if t.status == TrialStatus.COMPLETED)
        running = sum(1 for t in self.all_trials if t.status == TrialStatus.RUNNING)
        pending = sum(1 for t in self.all_trials if t.status == TrialStatus.PENDING)
        pruned = sum(1 for t in self.all_trials if t.status == TrialStatus.PRUNED)
        failed = sum(1 for t in self.all_trials if t.status == TrialStatus.FAILED)
        
        summary = f"""
{'='*60}
Hyperband Progress Summary
{'='*60}
Total Trials: {total}
  - Completed: {completed}
  - Running: {running}
  - Pending: {pending}
  - Pruned: {pruned}
  - Failed: {failed}

Brackets Progress:
"""
        for bracket in self.brackets:
            bracket_trials = [t for t in self.all_trials if t.bracket == bracket.bracket_id]
            bracket_completed = sum(1 for t in bracket_trials if t.status == TrialStatus.COMPLETED)
            summary += f"  Bracket {bracket.bracket_id}: Rung {bracket.current_rung}/{len(bracket.rungs)-1}, "
            summary += f"{bracket_completed} completed\n"
            
        if completed > 0:
            try:
                best_config, best_metric, _ = self.get_best_config()
                summary += f"\nBest {self.metric}: {best_metric:.4f}\n"
            except:
                pass
                
        summary += f"{'='*60}"
        return summary


class AdaptiveEarlyStopping:
    """
    Adaptive early stopping that considers multiple failure modes:
    1. Loss explosion (NaN or very large values)
    2. No improvement over patience epochs
    3. Metric degradation relative to other trials
    """
    
    def __init__(self,
                 patience: int = 20,
                 min_delta: float = 0.001,
                 divergence_threshold: float = 10.0,
                 relative_threshold: float = 2.0,
                 mode: str = "min"):
        """
        Args:
            patience: Epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            divergence_threshold: Loss multiplier to detect divergence
            relative_threshold: Multiple of best trial's metric to tolerate
            mode: "min" or "max"
        """
        self.patience = patience
        self.min_delta = min_delta
        self.divergence_threshold = divergence_threshold
        self.relative_threshold = relative_threshold
        self.mode = mode
        
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        self.initial_metric = None
        self.wait = 0
        self.stopped = False
        self.reason = ""
        
    def check(self, current_metric: float, 
              reference_best: Optional[float] = None) -> Tuple[bool, str]:
        """
        Check if training should stop.
        
        Args:
            current_metric: Current epoch's metric
            reference_best: Best metric from other trials (for relative comparison)
            
        Returns:
            Tuple of (should_stop, reason)
        """
        # Check for NaN or Inf
        if math.isnan(current_metric) or math.isinf(current_metric):
            self.stopped = True
            self.reason = "Metric is NaN or Inf"
            return True, self.reason
            
        # Initialize if first call
        if self.initial_metric is None:
            self.initial_metric = current_metric
            
        # Check for divergence (loss explosion)
        if self.mode == 'min':
            if current_metric > self.initial_metric * self.divergence_threshold:
                self.stopped = True
                self.reason = f"Divergence detected: {current_metric:.4f} > {self.initial_metric * self.divergence_threshold:.4f}"
                return True, self.reason
        else:
            if current_metric < self.initial_metric / self.divergence_threshold:
                self.stopped = True
                self.reason = f"Collapse detected: {current_metric:.4f}"
                return True, self.reason
                
        # Check relative to other trials
        if reference_best is not None:
            if self.mode == 'min':
                if current_metric > reference_best * self.relative_threshold:
                    self.stopped = True
                    self.reason = f"Far worse than best trial: {current_metric:.4f} vs {reference_best:.4f}"
                    return True, self.reason
            else:
                if current_metric < reference_best / self.relative_threshold:
                    self.stopped = True
                    self.reason = f"Far worse than best trial: {current_metric:.4f} vs {reference_best:.4f}"
                    return True, self.reason
                    
        # Standard patience check
        improved = False
        if self.mode == 'min':
            if current_metric < self.best_metric - self.min_delta:
                improved = True
        else:
            if current_metric > self.best_metric + self.min_delta:
                improved = True
                
        if improved:
            self.best_metric = current_metric
            self.wait = 0
        else:
            self.wait += 1
            
        if self.wait >= self.patience:
            self.stopped = True
            self.reason = f"No improvement for {self.patience} epochs"
            return True, self.reason
            
        return False, ""
    
    def reset(self):
        """Reset the early stopping state."""
        self.best_metric = float('inf') if self.mode == 'min' else float('-inf')
        self.initial_metric = None
        self.wait = 0
        self.stopped = False
        self.reason = ""
