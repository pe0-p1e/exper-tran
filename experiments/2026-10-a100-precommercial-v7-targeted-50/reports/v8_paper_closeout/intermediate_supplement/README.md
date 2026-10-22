# Complete numerical intermediate supplement

Includes prior exploratory and ablation measurements that were not already published. file_inventory.json.gz lists every scoped numeric file, its SHA256 and either an existing publication location or a member in this supplement. All source formats .json/.jsonl/.csv/.npy/.npz are inventoried, subject to the explicit privacy exclusions in coverage.json.

Reconstruct: `cat numeric_intermediates.tar.gz.part* > numeric_intermediates.tar.gz`, verify archive_sha256 from coverage.json, then `tar -xzf numeric_intermediates.tar.gz`. Existing published equivalents are not duplicated. Raw commercial traces/authentication and non-final image corpora stay local. The final images are separately packaged under ../final_images/. Failed and partial experiments keep their original status.
