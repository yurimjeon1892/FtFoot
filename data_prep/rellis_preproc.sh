
echo "Enter the path to rellis-3d dataset: (ex: /path/to/the/RELLIS-3D)"
read data_root
echo "data_root: $data_root"

echo "Enter the sequence number: (ex: 0 1 2 3 4)"
read seq
echo "sequence: $seq"

python ./data_prep/rellis/rellis_surface_normal_from_pcd.py --data_root $data_root --seqs $seq
python ./data_prep/rellis/rellis_foot_print.py --data_root $data_root --seqs $seq
python ./data_prep/rellis/rellis_super_pixel_from_rgb.py --data_root $data_root --seqs $seq