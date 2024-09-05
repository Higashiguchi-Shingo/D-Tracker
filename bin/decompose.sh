# dir setting
cd `dirname $0`
cd ..
cd src

python3 decompose.py    --dataset "pythonlib" \
                        --start_date "2010-01-01" \
                        --end_date "2022-12-31" \
                        --dk 3 \
                        --dl 2 \
                        --ds 1 \
                        --seasonal_period 52 \
                        --stl_period 26 \
                        --ablation_seasonal 0 \
                        --ablation_diffusion 0 \
                        --outlier 1 \
                        --ts 448 \
                        --te 552


