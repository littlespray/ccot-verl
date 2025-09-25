#!/usr/bin/env python3
"""
Resume model from Hugging Face Hub if local path doesn't exist.
Downloads the model from HF Hub to the specified local path.
"""

import os
import sys
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download, repo_exists, HfApi


def check_hf_repo_exists(repo_id: str, token: str) -> bool:
    """
    Check if a repository exists on Hugging Face Hub.
    
    Args:
        repo_id: Repository ID in format "username/repo_name"
        token: HF authentication token
    
    Returns:
        True if repo exists, False otherwise
    """
    try:
        return repo_exists(repo_id, token=token)
    except Exception as e:
        print(f"Error checking HF repo: {e}")
        return False


def get_latest_branch(repo_id: str, token: str) -> str:
    """
    Get the latest branch from HuggingFace repository.
    Branches are in format YYYYMMDD_HHMMSS, so alphabetical sort gives us the latest.
    
    Args:
        repo_id: Repository ID in format "username/repo_name"
        token: HF authentication token
        
    Returns:
        Latest branch name, or "main" if no timestamp branches found
    """
    api = HfApi(token=token)
    
    try:
        # Get all branches
        refs = api.list_repo_refs(repo_id=repo_id, token=token)
        
        # Find timestamp branches (format: YYYYMMDD_HHMMSS)
        timestamp_branches = []
        for branch in refs.branches:
            branch_name = branch.ref.replace("refs/heads/", "")
            # Check if it matches timestamp format
            if len(branch_name) == 15 and branch_name[8] == '_':
                try:
                    # Validate it's a valid timestamp format
                    date_part = branch_name[:8]
                    time_part = branch_name[9:]
                    if date_part.isdigit() and time_part.isdigit():
                        timestamp_branches.append(branch_name)
                except:
                    continue
        
        if timestamp_branches:
            # Sort alphabetically (works for YYYYMMDD_HHMMSS format)
            timestamp_branches.sort()
            latest_branch = timestamp_branches[-1]  # Get the last (newest) one
            print(f"Found {len(timestamp_branches)} checkpoint branches")
            print(f"Latest branch: {latest_branch}")
            return latest_branch
        else:
            print("No timestamp branches found, using main branch")
            return "main"
            
    except Exception as e:
        print(f"Error getting branches: {e}")
        print("Falling back to main branch")
        return "main"


def download_from_hf(repo_id: str, local_dir: str, token: str, branch: str = None):
    """
    Download model from Hugging Face Hub to local directory.
    Automatically downloads the latest checkpoint branch unless a specific branch is provided.
    
    Args:
        repo_id: Repository ID in format "username/repo_name"
        local_dir: Local directory to download to
        token: HF authentication token
        branch: Specific branch to download (optional, defaults to latest)
    """
    print(f"Downloading from Hugging Face Hub: {repo_id}")
    print(f"Target directory: {local_dir}")
    
    # Get the branch to download
    if branch:
        target_branch = branch
        print(f"Using specified branch: {target_branch}")
    else:
        target_branch = get_latest_branch(repo_id, token)
    
    try:
        # Create parent directories if they don't exist
        os.makedirs(os.path.dirname(local_dir), exist_ok=True)
        
        # Download the specific branch
        print(f"Downloading branch: {target_branch}")
        snapshot_download(
            repo_id=repo_id,
            revision=target_branch,  # Download specific branch
            local_dir=local_dir,
            token=token,
            local_dir_use_symlinks=False,  # Download actual files, not symlinks
        )
        
        print(f"Successfully downloaded {repo_id} (branch: {target_branch}) to {local_dir}")
        
    except Exception as e:
        print(f"Error downloading from HF: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Resume model from Hugging Face Hub if local path doesn't exist"
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Local path to check/download model to. The basename will be used as the HF repo name."
    )
    parser.add_argument(
        "--hf_user",
        type=str,
        default="sunshk",
        help="Hugging Face username (default: sunshk)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if local path exists"
    )
    parser.add_argument(
        "--branch",
        type=str,
        default=None,
        help="Specific branch to download (default: auto-detect latest)"
    )
    
    args = parser.parse_args()
    args.hf_token = os.getenv("HF_TOKEN")
    
    # Convert to absolute path
    input_path = os.path.abspath(args.input_path)
    
    # Check if local path exists
    if os.path.exists(input_path) and not args.force:
        print(f"Local path already exists: {input_path}")
        print("Use --force to re-download anyway")
        return
    
    # Extract repository name from path basename
    repo_name = os.path.basename(input_path)
    repo_id = f"{args.hf_user}/{repo_name}"
    
    print(f"Checking Hugging Face repository: {repo_id}")
    
    # Check if HF repo exists
    if not check_hf_repo_exists(repo_id, args.hf_token):
        print(f"Warning: Repository {repo_id} does not exist on Hugging Face Hub")
        print(f"Skipping download. Local path will not be created: {input_path}")
        return
    
    # Download from HF
    download_from_hf(repo_id, input_path, args.hf_token, args.branch)
    
    # Verify download
    if os.path.exists(input_path):
        # Check if directory has content
        if os.path.isdir(input_path) and os.listdir(input_path):
            print(f"Model successfully downloaded to: {input_path}")
        else:
            print(f"Warning: Downloaded directory appears to be empty: {input_path}")
    else:
        print(f"Error: Download completed but path not found: {input_path}")


if __name__ == "__main__":
    main()
