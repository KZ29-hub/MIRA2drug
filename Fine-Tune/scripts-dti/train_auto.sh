# biosnap-random (5, 10) 
cd /root/autodl-tmp/piccolo-embedding/scripts-dti/
epoch=1
neg=0
meta_path=/root/autodl-tmp/piccolo-embedding/meta_lists/dti_protein_biosnap_retrieval/data-fair/biosnap_random_pub_review_smile_fold1.txt
chechpoint_dir=/root/autodl-tmp/piccolo-embedding/scripts-dti/formal_biosnap_retri_8192/data-fair/biosnap_random_pub_review_smile_ep1_fold1_0410/
bash dti_ft_auto.sh $epoch $neg $meta_path $chechpoint_dir

# biosnap-random (5, 10)
cd /root/autodl-tmp/piccolo-embedding/scripts-dti/
epoch=1
neg=0
meta_path=/root/autodl-tmp/piccolo-embedding/meta_lists/dti_protein_biosnap_retrieval/data-fair/biosnap_random_pub_review_smile_fold2.txt
chechpoint_dir=/root/autodl-tmp/piccolo-embedding/scripts-dti/formal_biosnap_retri_8192/data-fair/biosnap_random_pub_review_smile_ep1_fold2_0410/
bash dti_ft_auto.sh $epoch $neg $meta_path $chechpoint_dir



# biosnap-random (5, 10) 
cd /root/autodl-tmp/piccolo-embedding/scripts-dti/
epoch=1
neg=0
meta_path=/root/autodl-tmp/piccolo-embedding/meta_lists/dti_protein_biosnap_retrieval/data-fair/biosnap_random_pub_wiki_fold1.txt
chechpoint_dir=/root/autodl-tmp/piccolo-embedding/scripts-dti/formal_biosnap_retri_8192/data-fair/biosnap_random_pub_wiki_ep1_fold1_0410/
bash dti_ft_auto.sh $epoch $neg $meta_path $chechpoint_dir


# biosnap-random (5, 10) 
cd /root/autodl-tmp/piccolo-embedding/scripts-dti/
epoch=1
neg=0
meta_path=/root/autodl-tmp/piccolo-embedding/meta_lists/dti_protein_biosnap_retrieval/data-fair/biosnap_random_pub_wiki_fold2.txt
chechpoint_dir=/root/autodl-tmp/piccolo-embedding/scripts-dti/formal_biosnap_retri_8192/data-fair/biosnap_random_pub_wiki_ep1_fold2_0410/
bash dti_ft_auto.sh $epoch $neg $meta_path $chechpoint_dir
