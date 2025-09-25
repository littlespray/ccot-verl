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


def delete_invalid_branches(api: HfApi, repo_id: str, token: str, branches: list) -> list:
    """
    Delete branches that don't have finish_check.txt file.
    Returns the list of valid branches after cleanup.
    
    Args:
        api: HfApi instance
        repo_id: Repository ID
        token: HF authentication token
        branches: List of branch names to check
        
    Returns:
        List of valid branches that contain finish_check.txt
    """
    valid_branches = []
    
    for branch in branches:
        if branch == "main":
            valid_branches.append(branch)
            continue
            
        try:
            # List files in the branch to check if finish_check.txt exists
            files = api.list_repo_files(
                repo_id=repo_id,
                revision=branch,
                token=token
            )
            
            # Check if finish_check.txt is in the files list
            if "finish_check.txt" in files:
                valid_branches.append(branch)
                print(f"Branch {branch}: Valid (contains finish_check.txt)")
            else:
                # File doesn't exist, delete the branch
                print(f"Branch {branch}: Invalid (missing finish_check.txt), deleting...")
                api.delete_branch(repo_id=repo_id, branch=branch, token=token)
                print(f"  ✗ Deleted branch: {branch}")
                
        except Exception as e:
            # Error accessing branch or listing files
            print(f"Branch {branch}: Error checking files: {e}")
            try:
                # Try to delete the problematic branch
                api.delete_branch(repo_id=repo_id, branch=branch, token=token)
                print(f"  ✗ Deleted problematic branch: {branch}")
            except Exception as del_e:
                print(f"  ! Failed to delete branch {branch}: {del_e}")
                # Still don't include it in valid branches
    
    return valid_branches

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
    
    - Uploads only once to a timestamped branch
    - Updates main branch reference without re-uploading
    - Maintains max 5 historical versions
    
    Args:
        checkpoint_path: Path to the checkpoint folder (e.g., input_path/global_step_100)
        repo_name: Repository name on HuggingFace
        step: The step number for this checkpoint
        hf_user: Hugging Face username
        hf_token: Hugging Face API token
    """
    folder_path = Path(checkpoint_path)
    if not folder_path.exists():
        print(f"Warning: Checkpoint folder {checkpoint_path} does not exist")
        return
    
    repo_id = f"{hf_user}/{repo_name}"
    api = HfApi(token=hf_token)
    
    try:
        # Ensure repository exists
        try:
            api.repo_info(repo_id=repo_id, token=hf_token)
        except:
            print(f"Creating repository: {repo_id}")
            create_repo(repo_id=repo_id, token=hf_token, private=False, exist_ok=True)
        
        # Generate unique branch name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        branch_name = f"{timestamp}"
        
        print(f"Uploading checkpoint to {repo_id} on branch: {branch_name}")
        
        # Create branch and upload in one operation
        api.create_branch(repo_id=repo_id, branch=branch_name, token=hf_token)
        
        # Upload latest_checkpointed_iteration.txt to root directory
        iteration_file = folder_path.parent / "latest_checkpointed_iteration.txt"
        api.upload_file(
            path_or_fileobj=str(iteration_file),
            path_in_repo="latest_checkpointed_iteration.txt",
            repo_id=repo_id,
            token=hf_token,
            revision=branch_name,
            commit_message=f"Update latest_checkpointed_iteration.txt for step {step}"
        )

        # Upload checkpoint folder
        upload_folder(
            folder_path=str(checkpoint_path),
            repo_id=repo_id,
            repo_type="model",
            path_in_repo=folder_path.name,
            token=hf_token,
            revision=branch_name,
            commit_message=f"Step {step}: Checkpoint upload"
        )
        
        # Upload finish_check.txt to mark this branch as complete
        finish_check_content = f"Checkpoint upload completed at {datetime.now().isoformat()}\nStep: {step}\n"
        api.upload_file(
            path_or_fileobj=finish_check_content.encode(),
            path_in_repo="finish_check.txt",
            repo_id=repo_id,
            token=hf_token,
            revision=branch_name,
            commit_message="Mark checkpoint upload as complete"
        )

        print(f"Uploaded to: https://huggingface.co/{repo_id}/tree/{branch_name}")
        
        # Clean up old branches
        _cleanup_old_branches(api, repo_id, hf_token)
                
    except Exception as e:
        print(f"Error during upload: {e}")


def _cleanup_old_branches(api: HfApi, repo_id: str, token: str, keep_count: int = 3):
    """Keep only the latest N branches."""
    try:
        refs = api.list_repo_refs(repo_id=repo_id, token=token)
        
        # Find and sort timestamp branches
        timestamp_branches = []
        for branch in refs.branches:
            name = branch.ref.replace("refs/heads/", "")
            if name != "main":
                # Try to parse as timestamp format YYYYMMDD_HHMMSS
                try:
                    # Validate timestamp format
                    if len(name) == 15 and name[8] == '_':
                        timestamp_branches.append(name)
                except:
                    continue
        


        # timestamp_branches = delete_invalid_branches(api, repo_id, token, timestamp_branches)
        # Sort by timestamp (branch names are sortable as strings)
        timestamp_branches.sort()

        
        # Delete old branches
        if len(timestamp_branches) > keep_count:
            to_delete = timestamp_branches[:-keep_count]
            print(f"Cleaning up {len(to_delete)} old branches...")
            
            for branch_name in to_delete:
                try:
                    api.delete_branch(repo_id=repo_id, branch=branch_name, token=token)
                    print(f"  ✗ Deleted: {branch_name}")
                except Exception as e:
                    print(f"  ! Failed to delete {branch_name}: {e}")
                
    except Exception as e:
        print(f"Note: Branch cleanup skipped: {e}")


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
