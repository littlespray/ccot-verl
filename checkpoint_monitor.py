#!/usr/bin/env python3
"""
Checkpoint Monitor Script
Monitors latest_checkpointed_iteration.txt and uploads new checkpoints to HuggingFace
"""

import os
import sys
import time
import argparse
import json
from pathlib import Path
from datetime import datetime
from huggingface_hub import HfApi, create_repo, upload_folder


def delete_invalid_repositories(api: HfApi, timestamped_repos: list, token: str) -> list:
    """
    Delete repositories that don't have finish_check.txt file.
    Returns the list of valid repositories after cleanup.
    
    Args:
        api: HfApi instance
        timestamped_repos: List of tuples (timestamp, repo_id) to check
        token: HF authentication token
        
    Returns:
        List of valid repositories that contain finish_check.txt
    """
    valid_repos = []
    
    for timestamp, repo_id in timestamped_repos:
        try:
            # List files in the repository to check if finish_check.txt exists
            files = api.list_repo_files(
                repo_id=repo_id,
                token=token
            )
            
            # Check if finish_check.txt is in the files list
            if "finish_check.txt" in files:
                valid_repos.append((timestamp, repo_id))
                print(f"Repository {repo_id}: Valid (contains finish_check.txt)")
            else:
                # File doesn't exist, delete the repository
                print(f"Repository {repo_id}: Invalid (missing finish_check.txt), deleting...")
                api.delete_repo(repo_id=repo_id, token=token)
                print(f"  ✗ Deleted repository: {repo_id}")
                
        except Exception as e:
            # Error accessing repository or listing files
            print(f"Repository {repo_id}: Error checking files: {e}")
            try:
                # Try to delete the problematic repository
                api.delete_repo(repo_id=repo_id, token=token)
                print(f"  ✗ Deleted problematic repository: {repo_id}")
            except Exception as del_e:
                print(f"  ! Failed to delete repository {repo_id}: {del_e}")
                # Still don't include it in valid repositories
    
    return valid_repos

def read_latest_step(input_path: Path) -> int:
    """Read the latest step from latest_checkpointed_iteration.txt"""
    iteration_file = input_path / "latest_checkpointed_iteration.txt"
    try:
        if iteration_file.exists():
            with open(iteration_file, 'r') as f:
                content = f.read().strip()
                return int(content)
    except Exception as e:
        print(f"Error reading iteration file: {e}")
    return -1


def upload_checkpoint_to_hf(checkpoint_path: str, repo_name: str, step: int, hf_user: str, hf_token: str):
    """
    Upload a checkpoint folder to Hugging Face Hub efficiently.
    
    - Creates a new timestamped repository for each checkpoint
    - Maintains max 5 historical versions
    
    Args:
        checkpoint_path: Path to the checkpoint folder (e.g., input_path/global_step_100)
        repo_name: Base repository name on HuggingFace
        step: The step number for this checkpoint
        hf_user: Hugging Face username
        hf_token: Hugging Face API token
    """
    folder_path = Path(checkpoint_path)
    if not folder_path.exists():
        print(f"Warning: Checkpoint folder {checkpoint_path} does not exist")
        return
    
    # Generate timestamped repository name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_repo_name = f"{repo_name}-{timestamp}"
    repo_id = f"{hf_user}/{timestamped_repo_name}"
    api = HfApi(token=hf_token)
    
    try:
        # Create new repository
        print(f"Creating new repository: {repo_id}")
        create_repo(repo_id=repo_id, token=hf_token, private=False, exist_ok=True)
        
        print(f"Uploading checkpoint to {repo_id}")
        
        # Upload latest_checkpointed_iteration.txt to root directory
        iteration_file = folder_path.parent / "latest_checkpointed_iteration.txt"
        api.upload_file(
            path_or_fileobj=str(iteration_file),
            path_in_repo="latest_checkpointed_iteration.txt",
            repo_id=repo_id,
            token=hf_token,
            commit_message=f"Update latest_checkpointed_iteration.txt for step {step}"
        )

        # Upload checkpoint folder
        upload_folder(
            folder_path=str(checkpoint_path),
            repo_id=repo_id,
            repo_type="model",
            path_in_repo=folder_path.name,
            token=hf_token,
            commit_message=f"Step {step}: Checkpoint upload"
        )
        
        # Upload finish_check.txt to mark this repository as complete
        finish_check_content = f"Checkpoint upload completed at {datetime.now().isoformat()}\nStep: {step}\n"
        api.upload_file(
            path_or_fileobj=finish_check_content.encode(),
            path_in_repo="finish_check.txt",
            repo_id=repo_id,
            token=hf_token,
            commit_message="Mark checkpoint upload as complete"
        )

        print(f"Uploaded to: https://huggingface.co/{repo_id}")
        
        # Clean up old repositories
        _cleanup_old_repositories(hf_user, repo_name, hf_token)
                
    except Exception as e:
        print(f"Error during upload: {e}")


def _cleanup_old_repositories(hf_user: str, base_repo_name: str, token: str, keep_count: int = 3):
    """Keep only the latest N timestamped repositories."""
    api = HfApi(token=token)
    
    try:
        # List all repositories for the user
        repos = api.list_models(author=hf_user, token=token)
        
        # Find timestamped repositories matching pattern
        timestamped_repos = []
        pattern_prefix = f"{base_repo_name}-"
        
        for repo in repos:
            repo_name = repo.modelId.split('/')[-1]  # Get repo name without user prefix
            if repo_name.startswith(pattern_prefix):
                # Extract timestamp suffix
                suffix = repo_name[len(pattern_prefix):]
                # Validate timestamp format YYYYMMDD_HHMMSS
                if len(suffix) == 15 and suffix[8] == '_':
                    try:
                        date_part = suffix[:8]
                        time_part = suffix[9:]
                        if date_part.isdigit() and time_part.isdigit():
                            timestamped_repos.append((suffix, repo.modelId))
                    except:
                        continue
        
        # Sort by timestamp
        timestamped_repos.sort(key=lambda x: x[0])
        
        # Delete old repositories
        if len(timestamped_repos) > keep_count:
            to_delete = timestamped_repos[:-keep_count]
            print(f"Cleaning up {len(to_delete)} old repositories...")
            
            for timestamp, repo_id in to_delete:
                try:
                    api.delete_repo(repo_id=repo_id, token=token)
                    print(f"  ✗ Deleted repository: {repo_id}")
                except Exception as e:
                    print(f"  ! Failed to delete {repo_id}: {e}")
                
    except Exception as e:
        print(f"Note: Repository cleanup skipped: {e}")


def monitor_checkpoints(input_path: str, hf_user: str, hf_token: str, check_interval: int = 300):
    """
    Monitor for new checkpoints and upload them to HuggingFace
    
    Args:
        input_path: Path to monitor for checkpoints
        hf_user: HuggingFace username
        hf_token: HuggingFace API token
        check_interval: Interval in seconds between checks (default: 300 = 5 minutes)
    """
    input_path = Path(input_path).resolve()
    repo_name = input_path.name  # Use basename of input_path as repo name
    
    print(f"Starting checkpoint monitor...")
    print(f"Input path: {input_path}")
    print(f"Repository: {hf_user}/{repo_name}")
    print(f"Check interval: {check_interval} seconds")
    print(f"Monitoring file: {input_path}/latest_checkpointed_iteration.txt")
    print("-" * 50)
    
    last_step = read_latest_step(input_path)
    print(f"Initial step: {last_step}")
    
    try:
        while True:
            # Read current step
            current_step = read_latest_step(input_path)
            
            # Check if step has changed
            if current_step > last_step and current_step != -1:
                print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] New checkpoint detected!")
                print(f"Step changed: {last_step} -> {current_step}")
                
                # Construct checkpoint folder path
                checkpoint_folder = input_path / f"global_step_{current_step}"
                
                # Upload the checkpoint
                upload_checkpoint_to_hf(
                    checkpoint_path=str(checkpoint_folder),
                    repo_name=repo_name,
                    step=current_step,
                    hf_user=hf_user,
                    hf_token=hf_token
                )
                
                last_step = current_step
                print(f"Updated last_step to: {last_step}")
            # else:
            #    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] No change detected (current step: {current_step})")
            
            # Wait for next check
            time.sleep(check_interval)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
    except Exception as e:
        print(f"\nError in monitoring loop: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Monitor and upload checkpoints to HuggingFace")
    parser.add_argument("--input_path", help="Path to monitor for checkpoints")
    parser.add_argument("--hf_user", default="sunshk", help="HuggingFace username (default: sunshk)")
    parser.add_argument("--interval", type=int, default=60, 
                       help="Check interval in seconds (default: 60 = 1 minutes)")
    
    args = parser.parse_args()
    args.hf_token = os.getenv("HF_TOKEN")
    
    # Validate input path
    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"Warning: Input path '{args.input_path}' does not exist")
    
    # Start monitoring
    monitor_checkpoints(
        input_path=args.input_path,
        hf_user=args.hf_user,
        hf_token=args.hf_token,
        check_interval=args.interval
    )


if __name__ == "__main__":
    main()
