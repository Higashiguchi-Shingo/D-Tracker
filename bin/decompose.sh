# dir setting
cd `dirname $0`
cd ..
cd src

# run ModelEstimation
dataset="device"
ablation_s=0
ablation_d=0
outlier=1
stl_period=26

dk=3
dl=2
ds=1

ts=448
te=552

python3 decompose.py    --dataset $dataset \
                        --start_date "2010-01-01" \
                        --end_date "2022-12-31" \
                        --dk $dk \
                        --dl $dl \
                        --ds $ds \
                        --seasonal_period 52 \
                        --stl_period $stl_period \
                        --ablation_seasonal $ablation_s \
                        --ablation_diffusion $ablation_d \
                        --outlier $outlier \
                        --ts $ts \
                        --te $te


