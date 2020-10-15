

def get_target_distribution(data,batch_size,n_rows):
    samples=[]
    sample =enumerate(data)
    n_iters=int(n_rows/batch_size)+1
    for i in range(n_iters):
        _ , (_ , sample_targets) = next(sample)
        samples.extend(sample_targets.tolist())

    return samples