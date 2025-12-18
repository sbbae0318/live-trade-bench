"""
Parquet 파일명에서 live_YYYYMMDDTHHMMSS_ prefix를 제거하는 스크립트
"""
import os
from pathlib import Path
import re

def rename_parquet_files(data_dir: str):
    """
    지정된 디렉토리에서 모든 parquet 파일의 이름에서 live_YYYYMMDDTHHMMSS_ prefix를 제거합니다.
    
    Args:
        data_dir: parquet 파일이 있는 디렉토리 경로
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"Error: Directory {data_dir} does not exist")
        return
    
    # 모든 하위 디렉토리에서 parquet 파일 찾기
    parquet_files = list(data_path.rglob("*.parquet"))
    
    if not parquet_files:
        print(f"No parquet files found in {data_dir}")
        return
    
    print(f"Found {len(parquet_files)} parquet files")
    
    renamed_count = 0
    skipped_count = 0
    
    # live_YYYYMMDDTHHMMSS_ 패턴 정규식
    pattern = re.compile(r'^live_\d{8}T\d{6}_(.+)$')
    
    for file_path in sorted(parquet_files):
        # realtime 디렉토리는 제외
        if "realtime" in file_path.parts:
            continue
        
        file_name = file_path.name
        file_stem = file_path.stem
        file_suffix = file_path.suffix
        
        # live_YYYYMMDDTHHMMSS_ prefix가 있는지 확인
        match = pattern.match(file_stem)
        
        if match:
            # prefix 제거된 새 파일명
            new_stem = match.group(1)
            new_name = new_stem + file_suffix
            new_path = file_path.parent / new_name
            
            # 새 파일명이 이미 존재하는지 확인
            if new_path.exists():
                print(f"⚠️  Skipping {file_name} -> {new_name} (target already exists)")
                skipped_count += 1
                continue
            
            try:
                # 파일명 변경
                file_path.rename(new_path)
                print(f"✓ Renamed: {file_name} -> {new_name}")
                renamed_count += 1
            except Exception as e:
                print(f"✗ Error renaming {file_name}: {e}")
        else:
            # prefix가 없으면 스킵
            skipped_count += 1
    
    print(f"\nSummary:")
    print(f"  Renamed: {renamed_count} files")
    print(f"  Skipped: {skipped_count} files")
    print(f"  Total: {len(parquet_files)} files")


if __name__ == "__main__":
    import sys
    
    # 기본 경로 설정
    base_dir = os.getenv("BASE_DIR", os.getcwd())
    default_data_dir = str(Path(base_dir) / "live_20251213T074350")
    
    # 명령줄 인자로 경로 지정 가능
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = default_data_dir
    
    print(f"Renaming parquet files in: {data_dir}")
    print("=" * 60)
    
    # 확인 메시지 (--yes 플래그가 있으면 스킵)
    if "--yes" not in sys.argv:
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled.")
            sys.exit(0)
    
    rename_parquet_files(data_dir)

