import json
from numpy import std


def analyze(attacks_file):
    """
    Compute statistics from the passed attacks_file

    Parameters
    ----------
    attacks_file : JSON file containing the adversarial attacks
    """

    with open(attacks_file) as f:
        results = json.load(f)

    num_posts = len(results)
    top_k = len(list(results.values())["top_k_attacks"])
    text_length_list = []
    vulnerable_text_length_list = []
    resistant_text_length_list = []
    num_successful_list = []
    num_attacks_list = []
    success_rate_list = []
    normalized_num_successful_list = []
    normalized_success_rate_list = []
    

    for result in results.values():
        text_length = result["stats"]["text_length"]
        text_length_list.append(text_length)

        num_successful = result["stats"]["num_successful"]
        num_successful_list.append(num_successful)
        if num_successful:  # if more than 1 successful attack
            vulnerable_text_length_list.append(text_length)
        else:
            resistant_text_length_list.append(text_length)

        num_attacks = result["stats"]["num_attacks"]
        num_attacks_list.append(num_attacks)

        success_rate = result["stats"]["success_rate"]
        success_rate_list.append(success_rate)

        normalized_num_successful_list.append(num_successful / text_length)
        normalized_success_rate_list.append(success_rate / text_length)

    num_vulnerable = len(vulnerable_text_length_list)
    num_resistant = len(resistant_text_length_list)

    num_successful_list_top_k = [min(5, num) for num in num_successful_list]

    mean_text_length, std_text_length = mean_and_std(text_length_list)
    mean_num_attacks, std_num_attacks = mean_and_std(num_attacks_list)
    mean_vulnerable_text_length, std_vulnerable_text_length = mean_and_std(vulnerable_text_length_list)
    mean_resistant_text_length, std_resistant_text_length = mean_and_std(resistant_text_length_list)
    mean_num_successful, std_num_successful = mean_and_std(num_successful_list)
    mean_success_rate, std_success_rate = mean_and_std(success_rate_list)
    mean_normalized_num_successful, std_normalized_num_successful = mean_and_std(normalized_num_successful_list)
    mean_num_successful_top_k, std_num_successful_top_k = mean_and_std(num_successful_list_top_k)
    mean_normalized_success_rate, std_normalized_success_rate = mean_and_std(normalized_success_rate_list)

    print(f"Number of posts (correctly classified as abusive): {num_posts}")
    print(f"Number vulnerable posts: {num_vulnerable}")
    print(f"Number resistant posts: {num_resistant}")
    print("Mean text length: {:.4f} +- {:.4f}".format(mean_text_length, std_text_length))
    print("Mean number of attacks: {:.4f} +- {:.4f}".format(mean_num_attacks, std_num_attacks))
    print("Mean number of successful attacks: {:.4f} +- {:.4f}".format(mean_num_successful, std_num_successful))
    print("Mean vulnerable text length: {:.4f} +- {:.4f}".format(mean_vulnerable_text_length, std_vulnerable_text_length))
    print("Mean resistant text length: {:.4f} +- {:.4f}".format(mean_resistant_text_length, std_resistant_text_length))
    print("Mean number of successful attacks: {:.4f} +- {:.4f}".format(mean_num_successful, std_num_successful))
    print("Mean number of successful attacks (normalized for text length): {:.4f} +- {:.4f}".format(mean_normalized_num_successful, std_normalized_num_successful))
    print("Mean number of successful attacks (in top {}): {:.4f} +- {:.4f}".format(top_k, mean_num_successful_top_k, std_num_successful_top_k))
    print("Mean success rate: {:.4f} +- {:.4f}".format(mean_success_rate, std_success_rate))
    print("Mean success rate (normalized for text length): {:.4f} +- {:.4f}".format(mean_normalized_success_rate, std_normalized_success_rate))


def mean_and_std(l):
    mean = sum(l) / len(l)
    standard_deviation = std(l)
    return mean, standard_deviation


def main():
    attacks_file = "data/adversarial_attacks_no-letters.json"
    analyze(attacks_file)


if __name__ == "__main__":
    main()
