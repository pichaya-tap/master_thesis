import torch
import time
# ... other necessary imports ...

from your_model_module import Generator  # Import your model class
# ... other necessary imports from your project ...

def test_model(model_g_path, device, test_loader, test_dir):
    # Load the model
    model_state_dict = torch.load(model_g_path)
    model = Generator()
    model.load_state_dict(model_state_dict)
    model = model.to(device)

    model.eval()

    test_passing_rates = []
    startTime = time.time()

    with torch.inference_mode():
        # Loop over the test dataset
        for batch_idx, (real,cond, water_tensor, density_tensor, data_name) in enumerate(test_loader): 
            # send the input to the device
            real = real.to(device) 
            cond = cond.to(device) 
            # Forward pass
            fake = gen(cond)

            plot_data(real.detach().cpu(), fake.detach().cpu(), data_name, test_dir)
            plot_slice(real.detach().cpu(), fake.detach().cpu(), density_tensor.detach().cpu(), data_name, test_dir)
            

            batch_passing_rates = cal_passing_rate(0.01, real.detach(), fake.detach())
            # Keep track of the passing_rate
            test_passing_rates += batch_passing_rates
        # Get average passing rate per epoch
        test_passing_rate, test_uncertainty = cal_epoch_passing_rate(test_passing_rates)

    endTime = time.time()
    print(f"Average Passing Rate: {test_passing_rate:.1f}")

    output_file_path = test_dir + "test_result.txt"
    with open(output_file_path, "w") as file:
        file.write("Test with generator model : {}\n".format(model_g_path))
        file.write("Test Passing Rate: {:.1f}% ± {:.1f}\n".format(test_passing_rate, test_uncertainty))
        file.write("Total number of samples in the test dataset: {}\n".format(len(test_subset)))
        file.write("Time taken to make one prediction: {:.2f}s\n".format((endTime - startTime)/len(test_subset)))

if __name__ == "__main__":
    # Define or load model_g_path, device, test_loader, test_dir
    test_model(model_g_path, device, test_loader, test_dir)
