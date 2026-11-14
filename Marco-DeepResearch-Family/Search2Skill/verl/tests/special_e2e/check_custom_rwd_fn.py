
import argparse

def check_congratulations_in_file(output_file):
    with open(output_file) as f:
        output = f.read()

    success_message = "Congratulations!!! You have called my_reward_function successfully!!!"
    assert success_message in output, f"Success message of my_reward_function not found in {output_file}"
    print("Check passes")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", required=True, type=str)

    args = parser.parse_args()

    check_congratulations_in_file(args.output_file)
