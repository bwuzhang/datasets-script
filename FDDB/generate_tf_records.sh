for i in 01 02 03 04 05 06 07 08 09 10
do
	python create_fddb_tf_record.py --data_dir=/local/mnt2/workspace2/chris/Databases/FDDB --FDDB_fold=FDDB-fold-$i --output_path=FDDB-fold-$i.record
done

python create_fddb_tf_record.py --data_dir=/local/mnt2/workspace2/chris/Databases/FDDB --output_path=FDDB-fold-all.record
