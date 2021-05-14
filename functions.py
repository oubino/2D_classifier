# functions

def print_metrics(metrics,  epoch_samples, phase): # not sure??
    print('Phase: Label, Value')
    for l in metrics.keys():
        output_per_landmark = []
        for k in metrics[l].keys():
            output_per_landmark.append("{}: {:4f}".format((l,k), metrics[l][k]/epoch_samples))
        print("{}: {}".format(phase, ", ".join(output_per_landmark)))