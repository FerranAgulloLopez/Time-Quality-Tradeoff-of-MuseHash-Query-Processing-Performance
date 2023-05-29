## Set up
- Install python3.9
- Install requirements.txt
- Set up data with this structure -> ...
- Create datasets -> ...
  - vgg16: `python3 create_dataset.py --dataset vgg16-features`
  - museHash: 
    - for each modality combination
      - for each length -> `python3 create_dataset.py --dataset muse-hash --additional-args "{\"bits\": NUMBER_BITS,\"modalities\":MODALITIES_ARRAY}"`
      - i.e. `python3 create_dataset.py --dataset muse-hash --additional-args "{\"bits\": 16,\"modalities\":[\"spatial\"]}"`

## RUN
`python3 run.py --dataset muse-hash-32-temporal --local --runs 1 --algorithm ball --count 30 --run-disabled --force`

## PLOT
`python3 plot.py --dataset muse-hash-32-temporal --count 30 --recompute -x muse-hash-precision`