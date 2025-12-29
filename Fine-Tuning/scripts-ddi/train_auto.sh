# cd /root/autodl-tmp/piccolo-embedding/scripts/
# epoch=30
# neg=1
# meta_path=/root/autodl-tmp/piccolo-embedding/meta_lists/tcm_n500/tcm_balanced_n500_fold1.txt
# chechpoint_dir=/root/autodl-tmp/piccolo-embedding/scripts/formal/tcm/tcm_balanced_n500_fold1_cls2_add1_stella_n1_epoch30_lr1e5_bs128_1130
# bash ft_auto.sh $epoch $neg $meta_path $chechpoint_dir

# cd /root/autodl-tmp/piccolo-embedding/scripts/
# epoch=30
# neg=1
# meta_path=/root/autodl-tmp/piccolo-embedding/meta_lists/tcm_n500/tcm_balanced_n500_fold2.txt
# chechpoint_dir=/root/autodl-tmp/piccolo-embedding/scripts/formal/tcm/tcm_balanced_n500_fold2_cls2_add1_stella_n1_epoch30_lr1e5_bs128_1130
# bash ft_auto.sh $epoch $neg $meta_path $chechpoint_dir

cd /root/autodl-tmp/piccolo-embedding/scripts/
epoch=20
neg=1
meta_path=/root/autodl-tmp/piccolo-embedding/meta_lists/tcm_n500/tcm_balanced_noadd_n500_fold1.txt
chechpoint_dir=/root/autodl-tmp/piccolo-embedding/scripts/formal/tcm/tcm_balanced_n500_fold1_cls2_noadd1_stella_n1_epoch20_lr1e5_bs128_1130-test
bash ft_auto.sh $epoch $neg $meta_path $chechpoint_dir

cd /root/autodl-tmp/piccolo-embedding/scripts/
epoch=20
neg=1
meta_path=/root/autodl-tmp/piccolo-embedding/meta_lists/tcm_n500/tcm_balanced_noadd_n500_fold2.txt
chechpoint_dir=/root/autodl-tmp/piccolo-embedding/scripts/formal/tcm/tcm_balanced_n500_fold2_cls2_noadd1_stella_n1_epoch20_lr1e5_bs128_1130
bash ft_auto.sh $epoch $neg $meta_path $chechpoint_dir