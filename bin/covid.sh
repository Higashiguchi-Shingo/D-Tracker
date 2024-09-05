# dir setting
cd `dirname $0`
cd ..
cd src

# run D-Tracker
dataset="covid"
lc=56
lf=21
init=0
rankupdate=1
ablation_s=0
ablation_d=0
outlier=1

stl_period=13

python3 main.py --dataset $dataset \
                --lc $lc \
                --lf $lf \
                --init $init \
                --rankupdate $rankupdate \
                --outlier $outlier \
                --ablation_seasonal $ablation_s \
                --ablation_diffusion $ablation_d \
                --maxdk 2 \
                --start_date "2020-01-01" \
                --end_date "2022-09-30" \
                --seasonal_period 7 \
                --stl_period $stl_period



# compute forecasting accuracy
out_lf="7/14/21"
start_timestep=364
end_timestep=968

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