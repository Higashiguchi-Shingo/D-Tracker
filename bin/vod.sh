# dir setting
cd `dirname $0`
cd ..
cd src

# run D-Tracker
dataset="vod"
lc=104
lf=39
init=0
rankupdate=1
ablation_s=0
ablation_d=0
outlier=1

stl_period=52

python3 main.py --dataset $dataset \
                --lc $lc \
                --lf $lf \
                --init $init \
                --rankupdate $rankupdate \
                --outlier $outlier \
                --ablation_seasonal $ablation_s \
                --ablation_diffusion $ablation_d \
                --stl_period $stl_period



# compute forecasting accuracy
out_lf="13/26/39"
start_timestep=312
end_timestep=636

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