# dir setting
cd `dirname $0`
cd ..
cd src

dataset="covid"
lc=42

rankupdate=1
ablation_s=0
ablation_d=0
outlier=1

stl_period=26

<<comment
out_lf="13/26/39"
start_timestep=520
end_timestep=636


for dataset in $datasets
do
    python3 accuracy_viz.py --dataset $dataset \
                            --lc $lc \
                            --lf $out_lf \
                            --start_timestep $start_timestep \
                            --end_timestep $end_timestep \
                            --rankupdate $rankupdate \
                            --ablation_seasonal $ablation_s \
                            --ablation_diffusion $ablation_d \
                            --outlier $outlier
done
comment

out_lf="7/14/21"
start_timestep=366
end_timestep=966

python3 accuracy.py --dataset $dataset \
                    --lc $lc \
                    --lf $out_lf \
                    --start_timestep $start_timestep \
                    --end_timestep $end_timestep \
                    --rankupdate $rankupdate \
                    --ablation_seasonal $ablation_s \
                    --ablation_diffusion $ablation_d \
                    --outlier $outlier \
                    --stl_period $stl_period