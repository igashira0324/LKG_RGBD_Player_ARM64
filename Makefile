# LKG RGBD Operation Makefile

.PHONY: sync concat setup clean check-env

# Default Target
all: help

help:
	@echo "LKG RGBD Operation Commands:"
	@echo "  make sync    - Start synchronized multi-display playback"
	@echo "  make concat  - Combine individual segments into final display videos"
	@echo "  make clean   - Remove temporary sync/done signal files"
	@echo "  make check-env - Verify environment and required files"

sync: check-env
	bash run_lkg_sync_multi.sh

concat:
	bash concat_lkg_videos.sh

check-env:
	@echo "Checking environment..."
	@if [ ! -d "lkg-env" ]; then echo "Error: lkg-env not found. Run setup if needed."; exit 1; fi
	@if [ ! -f "lkg_rgbd_player.py" ]; then echo "Error: lkg_rgbd_player.py missing."; exit 1; fi
	@echo "Environment OK."

clean:
	rm -f /tmp/lkg_sync_go /tmp/lkg_p1_done /tmp/lkg_p2_done
	@echo "Cleaned temporary signal files."
