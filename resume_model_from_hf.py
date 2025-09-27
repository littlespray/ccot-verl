#!/usr/bin/env python3
"""
Resume model from Hugging Face Hub if local path doesn't exist.
Downloads the model from HF Hub to the specified local path.
"""

import os
import sys
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download, HfApi


def find_valid_timestamped_repos(api: HfApi, base_repo_name: str, hf_user: str, token: str) -> list:
    """
    Find valid timestamped repositories that contain finish_check.txt.
    
    Args:
        api: HfApi instance
        base_repo_name: Base repository name (without timestamp)
        hf_user: HuggingFace username
        token: HF authentication token
        
    Returns:
        List of tuples (timestamp, repo_id) for valid repositories, sorted by timestamp
    """
    valid_repos = []
    
    try:
        # List all repositories for the user
        repos = api.list_models(author=hf_user, token=token)
        
        # Find timestamped repositories matching pattern
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
                            # Check if repository has finish_check.txt
                            try:
                                files = api.list_repo_files(
                                    repo_id=repo.modelId,
                                    token=token
                                )
                                if "finish_check.txt" in files:
                                    valid_repos.append((suffix, repo.modelId))
                                    print(f"Found valid repository: {repo.modelId}")
                                else:
                                    print(f"Repository {repo.modelId}: Missing finish_check.txt, skipping")
                            except Exception as e:
                                print(f"Error checking repository {repo.modelId}: {e}")
                    except:
                        continue
        
        # Sort by timestamp
        valid_repos.sort(key=lambda x: x[0])
        
    except Exception as e:
        print(f"Error searching for repositories: {e}")
    
    return valid_repos


def get_latest_timestamped_repo(base_repo_name: str, hf_user: str, token: str) -> str:
    """
    Get the latest timestamped repository for a given base name.
    Repositories are in format {base_name}-YYYYMMDD_HHMMSS.
    
    Args:
        base_repo_name: Base repository name (without timestamp)
        hf_user: HuggingFace username
        token: HF authentication token
        
    Returns:
        Latest repository ID, or None if no timestamped repositories found
    """
    api = HfApi(token=token)
    
    # Find valid timestamped repositories
    valid_repos = find_valid_timestamped_repos(api, base_repo_name, hf_user, token)
    
    if valid_repos:
        # Get the latest (last in sorted list)
        latest_timestamp, latest_repo_id = valid_repos[-1]
        print(f"Found {len(valid_repos)} valid timestamped repositories")
        print(f"Latest repository: {latest_repo_id} (timestamp: {latest_timestamp})")
        return latest_repo_id
    else:
        print("No valid timestamped repositories found")
        return None


def download_from_hf(base_repo_name: str, hf_user: str, local_dir: str, token: str, specific_repo: str = None):
    """
    Download model from Hugging Face Hub to local directory.
    Automatically downloads the latest timestamped repository unless a specific one is provided.
    
    Args:
        base_repo_name: Base repository name (without timestamp)
        hf_user: HuggingFace username
        local_dir: Local directory to download to
        token: HF authentication token
        specific_repo: Specific repository ID to download (optional, defaults to latest)
    """
    # Determine which repository to download
    if specific_repo:
        target_repo_id = specific_repo
        print(f"Using specified repository: {target_repo_id}")
    else:
        target_repo_id = get_latest_timestamped_repo(base_repo_name, hf_user, token)
    
    if not target_repo_id:
        print("No available timestamped repositories found. Stop Resume.")
        return False
    
    print(f"Downloading from Hugging Face Hub: {target_repo_id}")
    print(f"Target directory: {local_dir}")
    
    try:
        # Create parent directories if they don't exist
        os.makedirs(os.path.dirname(local_dir), exist_ok=True)
        
        # Download the repository
        print(f"Downloading repository: {target_repo_id}")
        snapshot_download(
            repo_id=target_repo_id,
            local_dir=local_dir,
            token=token,
            local_dir_use_symlinks=False,  # Download actual files, not symlinks
        )
        
        print(f"Successfully downloaded {target_repo_id} to {local_dir}")
        
    except Exception as e:
        print(f"Error downloading from HF: {e}")
        return False
    
    return True


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
        help="Specific repository to download (full repo ID, default: auto-detect latest timestamped)"
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
    base_repo_name = os.path.basename(input_path)
    
    print(f"Searching for timestamped repositories with base name: {base_repo_name}")
    
    # Download from HF
    status = download_from_hf(base_repo_name, args.hf_user, input_path, args.hf_token, args.branch)
    if not status:
        return
    
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
