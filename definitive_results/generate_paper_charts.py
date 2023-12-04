import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def __plot_lines(lines, y_label, x_label, x_ticks, title, path):
    fig, ax = plt.subplots(figsize=(40, 20))
    for line in lines:
        y, y_error, x, label = line
        ax.plot(x, y, label=label)
        ax.fill_between(x, y - y_error, y + y_error, alpha=0.3)
    plt.xlabel(x_label)
    plt.xticks(x_ticks)
    plt.ylabel(y_label)
    plt.title(title)
    ax.legend(loc='best', shadow=True)
    plt.savefig(path, bbox_inches='tight')


def speedup_data_vs_query_parallelism():
    def __compute_speedup(qps_mean: np.ndarray, qps_error: np.ndarray):
        baseline_mean = qps_mean[0]
        baseline_error = qps_error[0]
        speedup = np.divide(qps_mean, np.repeat([baseline_mean], qps_mean.shape[0], axis=0))
        relative_uncertainty_1 = np.divide(np.repeat([baseline_error], qps_error.shape[0], axis=0), np.repeat([baseline_mean], qps_mean.shape[0], axis=0))
        relative_uncertainty_2 = np.divide(qps_error, qps_mean)
        uncertainty = np.add(relative_uncertainty_1, relative_uncertainty_2)
        return speedup, uncertainty

    def __dict_compute_speedup(qps_dict: dict):
        speedup_dict = {}
        for hash_length in qps_dict.keys():
            qps_mean = np.asarray(qps_dict[hash_length]['mean'])
            qps_error = np.asarray(qps_dict[hash_length]['error'])
            speedup, uncertainty = __compute_speedup(qps_mean, qps_error)
            speedup_dict[hash_length] = {
                'mean': speedup,
                'error': uncertainty
            }
        return speedup_dict

    def __compute_inverted_speedup(speedup: np.ndarray, optimal: np.ndarray):
        return optimal + (optimal - speedup)

    def __dict_compute_inverted_speedup(speedup_dict: dict, optimal: np.ndarray):
        inverted_speedup_dict = {}
        for hash_length in speedup_dict.keys():
            speedup_mean = np.asarray(speedup_dict[hash_length]['mean'])
            speedup_error = np.asarray(speedup_dict[hash_length]['error'])
            inverted_speedup = __compute_inverted_speedup(speedup_mean, optimal)
            inverted_speedup_dict[hash_length] = {
                'mean': inverted_speedup,
                'error': speedup_error
            }
        return inverted_speedup_dict


    # initial data
    qps_query_parallelism_large = {
        '32': {
            'mean': [5.839611474993033, 11.73803489819996, 22.602373603022862, 43.6054667665494, 79.28433404274531, 125.38594293244792],
            'error': [0.05301108559528854, 0.06703610844555084, 0.3320641853526819, 0.37150289635174055, 0.7782272335225552, 1.2619964371339834]
        },
        '128': {
            'mean': [1.873244693686018, 3.4745643255582834, 6.97420137945232, 13.377104004853596, 24.69289599950037, 42.0294493014437],
            'error': [0.010735223090178669, 0.2694574970936735, 0.20783053324905196, 0.1710241834221909, 0.2979947372354048, 1.2613417811195573]
        },
        '512': {
            'mean': [0.4971154043701028, 0.9135395756355082, 1.800633601427093, 3.4783933376290515, 6.689480853061815, 12.074821636789443],
            'error': [0.0064679272563895, 0.08075122438214992, 0.09798909170785088, 0.20499759812721116, 0.1046200119567212, 0.2625289557038015]
        },
        '2048': {
            'mean': [0.12553415261935685, 0.23226792692384532, 0.48425144379962803, 0.8754517172937405, 1.6536786291795174, 3.1755866447880683],
            'error': [0.0027742345216329894, 0.018847944715683077, 0.006104361956918636, 0.07195786726247441, 0.07746223181803635, 0.030061249123360476]
        }
    }
    qps_data_parallelism_large = {
        '32': {
            'mean': [5.515293165601007, 7.939625293754284, 8.188434078685102, 7.9685249430242235, 7.766278595330436, 7.686237811770226],
            'error': [0.7407900930188465, 0.1873622483601287, 0.020218533012516533, 0.14681229288006525, 0.23674973921907502, 0.046573980303216946]
        },
        '128': {
            'mean': [1.8443371429971145, 2.5829581776724493, 3.838360504226343, 7.8391130393196375, 8.139953446191564, 7.924960014750941],
            'error': [0.014902467747296831, 0.5255687178606618, 0.616288447304674, 0.05756357995619391, 0.006339599693868834, 0.04719968029353195]
        },
        '512': {
            'mean': [0.45926004964440004, 0.8606524630210861, 1.4084852050792385, 2.8359094439713344, 4.304573580512481, 4.401100506262982],
            'error': [0.04325538140408267, 0.12810948680697637, 0.13385550682349165, 0.061623347738147656, 0.012394206053574485, 0.0191137599583104]
        },
        '2048': {
            'mean': [0.11708961255719146, 0.2368951861413941, 0.37015639890725105, 0.6123277536825996, 0.9887565659327266, 1.7706781763923622],
            'error': [0.016070870633430236, 0.004416415958717977, 0.01799617677645341, 0.08322974760361541, 0.12269410169790138, 0.16903796335524063]
        }
    }
    qps_query_parallelism_aur = {
        '32': {
            'mean': [344.1591326777886, 669.8499878255259, 1143.745489799706, 1965.3464595656055, 3346.2376143848514, 5054.994088472898],
            'error': [0.33284600289534816, 8.011929377215958, 106.09121565379816, 471.38531809004115, 471.1130469543946, 66.2507973990292]
        },
        '128': {
            'mean': [119.84918995808007, 236.5313627387411, 437.96492030348816, 773.9942783523669, 1159.3026327263717, 1848.5668823905532],
            'error': [0.21589480099177497, 0.3192220678457302, 8.975281583329293, 54.92982952514066, 15.242358785127161, 94.07212537164361]
        },
        '512': {
            'mean': [32.08709432305078, 63.44791805344876, 119.8885285364117, 217.78656714281573, 381.6120407110567, 608.5872498601453],
            'error': [0.2785519010746276, 0.2787975927804012, 0.7866550604769557, 4.07913880480757, 14.503414309692845, 35.855982946553816]
        },
        '2048': {
            'mean': [8.100334426542304, 11.801525836655335, 21.99342108730171, 41.033582899429554, 80.17300951961623, 157.55191291580024],
            'error': [0.09070412806227318, 0.9768390270267584, 0.920695335193338, 1.7445655560298778, 12.477207549554286, 17.881545013986326]
        }
    }
    qps_data_parallelism_aur = {
        '32': {
            'mean': [344.1497290947717, 9.540590667865462, 9.4051232445524, 9.409458201143856, 9.318234783119536, 9.096583551712536],
            'error': [1.4309847192493497, 0.022722957157712706, 0.04127720028631173, 0.03089163818141832, 0.0915063265276604, 0.05395322862981664]
        },
        '128': {
            'mean': [119.96002194594853, 9.559220880429123, 9.413334785299845, 9.338701999396191, 9.269055364487938, 9.080431571642675],
            'error': [0.33535731619096243, 0.02835474268013742, 0.0341072458113983, 0.028534861327673636, 0.06862947241190079, 0.1412106123842698]
        },
        '512': {
            'mean': [32.0569531212159, 9.55480182773454, 9.47635222651186, 9.378574365654448, 9.25143313987814, 9.146550998611346],
            'error': [0.2804596237986049, 0.07429677867389838, 0.03325425553550039, 0.06713261550828008,0.06983999360619786, 0.0669745751365776]
        },
        '2048': {
            'mean': [7.4831805898368895, 9.366043929645173, 9.523949536344608, 9.553609766986384, 9.361273999426514, 9.079554220971591],
            'error': [0.9504805425047387, 0.08411134951288438, 0.018909564340458603, 0.04620941501657367,0.07849197726711994, 0.03839289542587718]
        }
    }
    optimal_speedup = np.asarray([1, 2, 4, 8, 16, 32])

    # compute speedup
    speedup_query_parallelism_large = __dict_compute_speedup(qps_query_parallelism_large)
    speedup_data_parallelism_large = __dict_compute_speedup(qps_data_parallelism_large)
    speedup_query_parallelism_aur = __dict_compute_speedup(qps_query_parallelism_aur)
    speedup_data_parallelism_aur = __dict_compute_speedup(qps_data_parallelism_aur)

    # plot
    #inverted_speedup_data_parallelism_large = __dict_compute_inverted_speedup(speedup_data_parallelism_large, optimal_speedup)

    # colors = ['#8B8C89', '#274C77', '#6096BA', '#A3CEF1']
    colors = ['green', 'blue', 'orange', 'orchid']
    lines = [
        (
            speedup_data_parallelism_large[hash_length]['mean'],
            speedup_data_parallelism_large[hash_length]['error'],
            optimal_speedup,
            colors[index]
        )
        for index, hash_length in enumerate(speedup_data_parallelism_large.keys())
    ]

    fig, axs = plt.subplots(1, 3, sharey=True, figsize=(9 * 3, 6))
    fig.subplots_adjust(wspace=0)

    for line in lines:
        y, y_error, x, color = line
        axs[0].plot(x, y, color=color, marker='o')
        axs[0].fill_between(x, y - y_error, y + y_error, alpha=0.3, color=color)
    axs[0].plot(optimal_speedup, optimal_speedup, label='optimal scalability', color='#C1666B')
    axs[0].set_xlabel('number processes', fontsize=20)
    axs[0].set_xticks(optimal_speedup, optimal_speedup, fontsize=18)
    axs[0].set_title('(a) Data Parallelism & large synthetic', fontsize=23)
    axs[0].set_yticks(optimal_speedup[1:], optimal_speedup[1:], fontsize=18)
    axs[0].set_ylabel('Speed-up', fontsize=18)

    lines = [
        (
            speedup_query_parallelism_large[hash_length]['mean'],
            speedup_query_parallelism_large[hash_length]['error'],
            optimal_speedup,
            colors[index]
        )
        for index, hash_length in enumerate(speedup_query_parallelism_large.keys())
    ]

    for line in lines:
        y, y_error, x, color = line
        axs[1].plot(x, y, color=color, marker='o')
        axs[1].fill_between(x, y - y_error, y + y_error, alpha=0.3, color=color)
    axs[1].set_xlabel('number processes', fontsize=20)
    axs[1].set_xticks(optimal_speedup, optimal_speedup, fontsize=18)
    axs[1].plot(optimal_speedup, optimal_speedup, label='optimal scalability', color='#C1666B')
    axs[1].set_title('(b) Query Parallelism & large synthetic', fontsize=23)

    lines = [
        (
            speedup_query_parallelism_aur[hash_length]['mean'],
            speedup_query_parallelism_aur[hash_length]['error'],
            optimal_speedup,
            colors[index]
        )
        for index, hash_length in enumerate(speedup_query_parallelism_aur.keys())
    ]

    for line in lines:
        y, y_error, x, color = line
        axs[2].plot(x, y, color=color, marker='o')
        axs[2].fill_between(x, y - y_error, y + y_error, alpha=0.3, color=color)
    axs[2].plot(optimal_speedup, optimal_speedup, label='optimal scalability', color='#C1666B')
    axs[2].set_xlabel('number processes', fontsize=20)
    axs[2].set_xticks(optimal_speedup, optimal_speedup, fontsize=18)
    axs[2].set_title('(c) Query Parallelism & AU-AIR', fontsize=23)

    handles, labels = axs[0].get_legend_handles_labels()
    line_1 = Line2D([0], [0], label='32 hash length', color='green')
    line_2 = Line2D([0], [0], label='128 hash length', color='blue')
    line_3 = Line2D([0], [0], label='512 hash length', color='orange')
    line_4 = Line2D([0], [0], label='2048 hash length', color='orchid')
    handles.extend([line_1, line_2, line_3, line_4])
    axs[0].legend(handles=handles, loc='best', fontsize=22)
    plt.savefig(f'paper_charts/speedup_data_vs_query_parallelism.png', bbox_inches='tight')


def speedup_cpu_vs_gpu():
    def __compute_speedup(qps_mean: np.ndarray, qps_error: np.ndarray, baseline_mean: float, baseline_error: float):
        speedup = np.divide(qps_mean, np.repeat([baseline_mean], qps_mean.shape[0], axis=0))
        relative_uncertainty_1 = np.divide(np.repeat([baseline_error], qps_error.shape[0], axis=0), np.repeat([baseline_mean], qps_mean.shape[0], axis=0))
        relative_uncertainty_2 = np.divide(qps_error, qps_mean)
        uncertainty = np.add(relative_uncertainty_1, relative_uncertainty_2)
        return speedup, uncertainty

    def __dict_compute_speedup(qps_dict: dict):
        speedup_dict = {}
        for hash_length in qps_dict.keys():
            qps_mean = np.asarray(qps_dict[hash_length]['mean'])
            qps_error = np.asarray(qps_dict[hash_length]['error'])
            baseline_mean = qps_mean[0]
            baseline_error = qps_error[0]
            speedup, uncertainty = __compute_speedup(qps_mean, qps_error, baseline_mean, baseline_error)
            speedup_dict[hash_length] = {
                'mean': speedup,
                'error': uncertainty
            }
        return speedup_dict

    def __dict_compute_speedup_baseline(qps_dict: dict, baseline_dict: dict):
        speedup_dict = {}
        for hash_length in qps_dict.keys():
            qps_mean = np.asarray(qps_dict[hash_length]['mean'])
            qps_error = np.asarray(qps_dict[hash_length]['error'])
            baseline_mean = baseline_dict[hash_length]['mean']
            baseline_error = baseline_dict[hash_length]['error']
            speedup, uncertainty = __compute_speedup(qps_mean, qps_error, baseline_mean, baseline_error)
            speedup_dict[hash_length] = {
                'mean': speedup,
                'error': uncertainty
            }
        return speedup_dict

    # initial data
    qps_query_parallelism_large = {
        '32': {
            'mean': [5.839611474993033, 11.73803489819996, 22.602373603022862],
            'error': [0.05301108559528854, 0.06703610844555084, 0.3320641853526819]
        },
        '128': {
            'mean': [1.873244693686018, 3.4745643255582834, 6.97420137945232],
            'error': [0.010735223090178669, 0.2694574970936735, 0.20783053324905196]
        },
        '512': {
            'mean': [0.4971154043701028, 0.9135395756355082, 1.800633601427093],
            'error': [0.0064679272563895, 0.08075122438214992, 0.09798909170785088]
        },
        '2048': {
            'mean': [0.12553415261935685, 0.23226792692384532, 0.48425144379962803],
            'error': [0.0027742345216329894, 0.018847944715683077, 0.006104361956918636]
        }
    }
    qps_query_parallelism_large_gpu = {
        '32': {
            'mean': [21.104572589765752, 35.82074092904078, 56.81775255457736],
            'error': [0.6959737402052791, 0.5664425390677391, 0.1866026032771886]
        },
        '128': {
            'mean': [21.09609487083733, 36.36285172598711, 57.65622201884353],
            'error': [0.15123938852234747, 1.0572957592972148, 1.5799136359074588]
        },
        '512': {
            'mean': [19.734008523966025, 34.53825560612073, 56.27735132990076],
            'error': [1.2796876051427148, 0.37266598494173914, 0.4856490609004546]
        },
        '2048': {
            'mean': [18.14725961274277, 31.4474636485215, 52.70931686337642],
            'error': [0.3999890494492262, 0.6586767792542159, 0.4147863118450304]
        }
    }
    optimal_speedup = np.asarray([1, 2, 4])

    # compute speedup
    speedup_query_parallelism_large = __dict_compute_speedup(qps_query_parallelism_large)
    speedup_query_parallelism_large_gpu = __dict_compute_speedup_baseline(
        qps_query_parallelism_large_gpu,
        {
            '32': {
                'mean': 5.839611474993033,
                'error': 0.05301108559528854,
            },
            '128': {
                'mean': 1.873244693686018,
                'error': 0.010735223090178669,
            },
            '512': {
                'mean': 0.4971154043701028,
                'error': 0.0064679272563895,
            },
            '2048': {
                'mean': 0.12553415261935685,
                'error': 0.0027742345216329894
            }
        }
    )

    # plot
    colors = ['green', 'blue', 'orange', 'orchid']
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(9, 6 * 2))
    fig.subplots_adjust(hspace=0.1)

    lines = [
        (
            speedup_query_parallelism_large_gpu[hash_length]['mean'],
            speedup_query_parallelism_large_gpu[hash_length]['error'],
            optimal_speedup,
            colors[index]
        )
        for index, hash_length in enumerate(speedup_query_parallelism_large_gpu.keys())
    ]

    for line in lines:
        y, y_error, x, color = line
        axs[0].plot(x, y, color=color, marker='o')
        axs[0].fill_between(x, y - y_error, y + y_error, alpha=0.3, color=color)
    axs[0].plot(optimal_speedup, optimal_speedup, label='optimal scalability', color='#C1666B')
    axs[0].set_title('GPU', fontsize=18)
    axs[0].set_yticks([1, 64, 128, 384, 512], [1, 64, 128, 384, 512], fontsize=18)
    axs[0].set_ylabel('Speed-up', fontsize=18, labelpad=-18)

    lines = [
        (
            speedup_query_parallelism_large[hash_length]['mean'],
            speedup_query_parallelism_large[hash_length]['error'],
            optimal_speedup,
            colors[index]
        )
        for index, hash_length in enumerate(speedup_query_parallelism_large.keys())
    ]

    for line in lines:
        y, y_error, x, color = line
        axs[1].plot(x, y, color=color, marker='o')
        axs[1].fill_between(x, y - y_error, y + y_error, alpha=0.3, color=color)
    axs[1].plot(optimal_speedup, optimal_speedup, label='optimal scalability', color='#C1666B')
    plt.xlabel('number processes / GPUs', fontsize=16)
    plt.xticks(optimal_speedup, fontsize=18)
    axs[1].set_title('CPU', fontsize=18)
    axs[1].set_yticks(optimal_speedup, optimal_speedup, fontsize=18)
    axs[1].set_ylabel('Speed-up', fontsize=18)

    handles, labels = axs[0].get_legend_handles_labels()
    line_1 = Line2D([0], [0], label='32 hash length', color='green')
    line_2 = Line2D([0], [0], label='128 hash length', color='blue')
    line_3 = Line2D([0], [0], label='512 hash length', color='orange')
    line_4 = Line2D([0], [0], label='2048 hash length', color='orchid')
    handles.extend([line_1, line_2, line_3, line_4])
    axs[0].legend(handles=handles, loc='best', fontsize=18)
    axs[1].legend(handles=handles, loc='best', fontsize=18)
    plt.savefig(f'paper_charts/cpu_vs_gpu_speedup.png', bbox_inches='tight')


def qps_cpu_vs_gpu():
    def __compute_speedup(qps_mean: np.ndarray, qps_error: np.ndarray, baseline_mean: float, baseline_error: float):
        speedup = np.divide(qps_mean, np.repeat([baseline_mean], qps_mean.shape[0], axis=0))
        relative_uncertainty_1 = np.divide(np.repeat([baseline_error], qps_error.shape[0], axis=0), np.repeat([baseline_mean], qps_mean.shape[0], axis=0))
        relative_uncertainty_2 = np.divide(qps_error, qps_mean)
        uncertainty = np.add(relative_uncertainty_1, relative_uncertainty_2)
        return speedup, uncertainty

    def __dict_compute_speedup(qps_dict: dict):
        speedup_dict = {}
        for hash_length in qps_dict.keys():
            qps_mean = np.asarray(qps_dict[hash_length]['mean'])
            qps_error = np.asarray(qps_dict[hash_length]['error'])
            baseline_mean = qps_mean[0]
            baseline_error = qps_error[0]
            speedup, uncertainty = __compute_speedup(qps_mean, qps_error, baseline_mean, baseline_error)
            speedup_dict[hash_length] = {
                'mean': speedup,
                'error': uncertainty
            }
        return speedup_dict

    def __dict_compute_speedup_baseline(qps_dict: dict, baseline_dict: dict):
        speedup_dict = {}
        for hash_length in qps_dict.keys():
            qps_mean = np.asarray(qps_dict[hash_length]['mean'])
            qps_error = np.asarray(qps_dict[hash_length]['error'])
            baseline_mean = baseline_dict[hash_length]['mean']
            baseline_error = baseline_dict[hash_length]['error']
            speedup, uncertainty = __compute_speedup(qps_mean, qps_error, baseline_mean, baseline_error)
            speedup_dict[hash_length] = {
                'mean': speedup,
                'error': uncertainty
            }
        return speedup_dict

    # initial data
    qps_query_parallelism_large = {
        '32': {
            'mean': [5.839611474993033, 11.73803489819996, 22.602373603022862],
            'error': [0.05301108559528854, 0.06703610844555084, 0.3320641853526819]
        },
        '128': {
            'mean': [1.873244693686018, 3.4745643255582834, 6.97420137945232],
            'error': [0.010735223090178669, 0.2694574970936735, 0.20783053324905196]
        },
        '512': {
            'mean': [0.4971154043701028, 0.9135395756355082, 1.800633601427093],
            'error': [0.0064679272563895, 0.08075122438214992, 0.09798909170785088]
        },
        '2048': {
            'mean': [0.12553415261935685, 0.23226792692384532, 0.48425144379962803],
            'error': [0.0027742345216329894, 0.018847944715683077, 0.006104361956918636]
        }
    }
    qps_query_parallelism_large_gpu = {
        '32': {
            'mean': [21.104572589765752, 35.82074092904078, 56.81775255457736],
            'error': [0.6959737402052791, 0.5664425390677391, 0.1866026032771886]
        },
        '128': {
            'mean': [21.09609487083733, 36.36285172598711, 57.65622201884353],
            'error': [0.15123938852234747, 1.0572957592972148, 1.5799136359074588]
        },
        '512': {
            'mean': [19.734008523966025, 34.53825560612073, 56.27735132990076],
            'error': [1.2796876051427148, 0.37266598494173914, 0.4856490609004546]
        },
        '2048': {
            'mean': [18.14725961274277, 31.4474636485215, 52.70931686337642],
            'error': [0.3999890494492262, 0.6586767792542159, 0.4147863118450304]
        }
    }

    # plot
    colors = ['green', 'blue', 'orange', 'orchid']
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(9, 6 * 2))
    fig.subplots_adjust(hspace=0.1)

    lines = [
        (
            np.asarray(qps_query_parallelism_large_gpu[hash_length]['mean']) / 1000,
            np.asarray(qps_query_parallelism_large_gpu[hash_length]['error']) / 1000,
            np.asarray([1, 2, 4]),
            colors[index]
        )
        for index, hash_length in enumerate(qps_query_parallelism_large_gpu.keys())
    ]

    for line in lines:
        y, y_error, x, color = line
        axs[0].plot(x, y, color=color, marker='o')
        axs[0].fill_between(x, y - y_error, y + y_error, alpha=0.3, color=color)
    axs[0].set_title('GPU', fontsize=18)
    axs[0].set_yticks([0.02, 0.03, 0.04, 0.05, 0.06], [0.02, 0.03, 0.04, 0.05, 0.06], fontsize=18)
    axs[0].set_ylabel('Queries per second (1/s) in scale of 1000', fontsize=16)

    lines = [
        (
            np.asarray(qps_query_parallelism_large[hash_length]['mean']) / 1000,
            np.asarray(qps_query_parallelism_large[hash_length]['error']) / 1000,
            np.asarray([1, 2, 4]),
            colors[index]
        )
        for index, hash_length in enumerate(qps_query_parallelism_large.keys())
    ]

    for line in lines:
        y, y_error, x, color = line
        axs[1].plot(x, y, color=color, marker='o')
        axs[1].fill_between(x, y - y_error, y + y_error, alpha=0.3, color=color)
    plt.xlabel('number processes / GPUs', fontsize=16)
    plt.xticks([1, 2, 4], fontsize=18)
    axs[1].set_title('CPU', fontsize=18)
    axs[1].set_yticks([0.00, 0.01, 0.02], [0.00, 0.01, 0.02], fontsize=18)
    axs[1].set_ylabel('Queries per second (1/s) in scale of 1000', fontsize=16)

    handles, labels = axs[0].get_legend_handles_labels()
    line_1 = Line2D([0], [0], label='32 hash length', color='green')
    line_2 = Line2D([0], [0], label='128 hash length', color='blue')
    line_3 = Line2D([0], [0], label='512 hash length', color='orange')
    line_4 = Line2D([0], [0], label='2048 hash length', color='orchid')
    handles.extend([line_1, line_2, line_3, line_4])
    axs[0].legend(handles=handles, loc='best', fontsize=18)
    axs[1].legend(handles=handles, loc='best', fontsize=18)
    plt.savefig(f'paper_charts/cpu_vs_gpu_qps.png', bbox_inches='tight')


def speedup_pynndescent():
    def __compute_speedup(qps_mean: np.ndarray, qps_error: np.ndarray):
        baseline_mean = qps_mean[0]
        baseline_error = qps_error[0]
        speedup = np.divide(qps_mean, np.repeat([baseline_mean], qps_mean.shape[0], axis=0))
        relative_uncertainty_1 = np.divide(np.repeat([baseline_error], qps_error.shape[0], axis=0), np.repeat([baseline_mean], qps_mean.shape[0], axis=0))
        relative_uncertainty_2 = np.divide(qps_error, qps_mean)
        uncertainty = np.add(relative_uncertainty_1, relative_uncertainty_2)
        return speedup, uncertainty

    def __dict_compute_speedup(qps_dict: dict):
        speedup_dict = {}
        for hash_length in qps_dict.keys():
            qps_mean = np.asarray(qps_dict[hash_length]['mean'])
            qps_error = np.asarray(qps_dict[hash_length]['error'])
            speedup, uncertainty = __compute_speedup(qps_mean, qps_error)
            speedup_dict[hash_length] = {
                'mean': speedup,
                'error': uncertainty
            }
        return speedup_dict

    # initial data
    qps_pynndescent_large = {
        '32': {
            'mean': [11708.504448089661, 20537.37371965065, 29241.2999547011, 42334.76232351085, 46485.54408665109, 37677.44648543301],
            'error': [231.45319988448057, 181.7264644166744, 3437.3729166524895, 7773.263284144664, 8246.713424880782, 5170.5430405109]
        },
        '128': {
            'mean': [4042.2247382688497, 7325.271222197247, 11398.164074378426, 17221.245976870818, 25042.41572805847, 25264.65400541122],
            'error': [80.30552987306814, 612.1497544524102, 1158.4885785612641, 1304.1956099773295, 934.665226757488, 4731.56078254328]
        },
        '512': {
            'mean': [199.07459845906192, 373.2974505972261, 730.1110054144525, 1153.7202832850464, 2074.7504315517153, 2552.325589006622],
            'error': [2.87626509918476, 28.325168656333023, 14.835203184379443, 104.20974933187333, 140.95203795045558, 163.9218940719088]
        },
        '2048': {
            'mean': [2.704015756545677, 4.96676459033988, 8.915763616729771, 17.847596244520297, 30.778244062125292, np.NAN],
            'error': [0.055557669498237754, 0.7005019820421902, 0.9021221770263185, 0.9666050183029049, 0.7051901000138706,  np.NAN]
        }
    }
    qps_pynndescent_medium = {
        '32': {
            'mean': [12070.702026643616, 20904.553385451047, 31330.587018988437, 41983.04090667638, 48960.46170953484, 36058.48088210679],
            'error': [1176.5339319507978, 981.8403292832554, 2503.5821241696985, 2264.511735638832, 1910.8856215927196, 3491.178388982202]
        },
        '128': {
            'mean': [5458.210741973914, 9769.790812828993, 14529.448904313667, 20762.618388564977, 28068.788952722425, 31229.81718570504],
            'error': [69.21526313797237, 434.4947518366551, 2256.0244232676196, 855.9709670644645, 2112.762466727134, 2490.050131985578]
        },
        '512': {
            'mean': [266.4055354191918, 535.555599307899, 921.1501591010325, 1635.542009752808, 2596.9926105066, 3378.254358433479],
            'error': [2.158989619501935, 7.150925625789641, 56.16794904359552, 101.65202715669326, 257.09551087090273, 181.04872321485146]
        },
        '2048': {
            'mean': [10.073985427168708, 20.077597019225873, 35.890604891600304, 67.99115175603856, 115.7854857276667, 183.71657986598206],
            'error': [0.1108809731968728, 0.2854227926241098, 3.6465206021727323, 3.1594093367087583, 1.7993736103446847, 4.484628444063581]
        }
    }
    qps_pynndescent_small = {
        '32': {
            'mean': [15461.198234707184, 26978.86507084209, 37991.6156663422, 51120.29338578734, 46656.38171581918, 39908.454811703195],
            'error': [1623.079577957843, 3065.1915775737466, 3266.919938940566, 4719.672988987863, 6055.5149897556685, 6249.413927297603]
        },
        '128': {
            'mean': [7438.565222093566, 12980.61795019754, 17809.802370891513, 25621.011612314494, 32248.216884485493, 27299.67427662292],
            'error': [107.77480673432378, 984.7465572670238, 1352.650764972775, 1019.3903550628577, 4349.188466868843, 7148.291773568872]
        },
        '512': {
            'mean': [473.7048868262253, 858.4717005060654, 1472.1881110539136, 2305.4851901499346, 3532.990396875214, 4883.115168901937],
            'error': [3.7191469912130177, 131.85073700428183, 30.499390594144103, 95.27760436910843, 435.8183459380423, 383.2308704590296]
        },
        '2048': {
            'mean': [39.43405070484888, 78.17392242510873, 148.6213409483211, 272.84054509538794, 444.0934874633551, 594.2999235257943],
            'error': [0.41910937996912484, 1.0593504256845807, 1.3295547411513369, 2.7156827135746475, 17.449252259820195, 41.54549184133701]
        }
    }

    optimal_speedup = np.asarray([1, 2, 4, 8, 16, 32])

    # compute speedup
    speedup_pynndescent_large = __dict_compute_speedup(qps_pynndescent_large)
    speedup_pynndescent_medium = __dict_compute_speedup(qps_pynndescent_medium)
    speedup_pynndescent_small = __dict_compute_speedup(qps_pynndescent_small)

    # plot
    colors = ['green', 'blue', 'orange', 'orchid']
    lines = [
        (
            speedup_pynndescent_small[hash_length]['mean'],
            speedup_pynndescent_small[hash_length]['error'],
            optimal_speedup,
            colors[index]
        )
        for index, hash_length in enumerate(speedup_pynndescent_small.keys())
    ]

    fig, axs = plt.subplots(1, 3, sharey=True, figsize=(9 * 3, 6))
    fig.subplots_adjust(wspace=0)

    for line in lines:
        y, y_error, x, color = line
        axs[0].plot(x, y, color=color, marker='o')
        axs[0].fill_between(x, y - y_error, y + y_error, alpha=0.3, color=color)
    axs[0].plot(optimal_speedup, optimal_speedup, label='optimal scalability', color='#C1666B')
    axs[0].set_xlabel('number processes', fontsize=20)
    axs[0].set_xticks(optimal_speedup[1:], optimal_speedup[1:], fontsize=18)
    axs[0].set_title('(a) small synthetic', fontsize=23)
    axs[0].set_yticks(optimal_speedup[1:], optimal_speedup[1:], fontsize=18)
    axs[0].set_ylabel('Speed-up', fontsize=18)

    lines = [
        (
            speedup_pynndescent_medium[hash_length]['mean'],
            speedup_pynndescent_medium[hash_length]['error'],
            optimal_speedup,
            colors[index]
        )
        for index, hash_length in enumerate(speedup_pynndescent_medium.keys())
    ]

    for line in lines:
        y, y_error, x, color = line
        axs[1].plot(x, y, color=color, marker='o')
        axs[1].fill_between(x, y - y_error, y + y_error, alpha=0.3, color=color)
    axs[1].set_xlabel('number processes', fontsize=20)
    axs[1].set_xticks(optimal_speedup[1:], optimal_speedup[1:], fontsize=18)
    axs[1].plot(optimal_speedup, optimal_speedup, label='optimal scalability', color='#C1666B')
    axs[1].set_title('(b) medium synthetic', fontsize=23)

    lines = [
        (
            speedup_pynndescent_large[hash_length]['mean'],
            speedup_pynndescent_large[hash_length]['error'],
            optimal_speedup,
            colors[index]
        )
        for index, hash_length in enumerate(speedup_pynndescent_large.keys())
    ]

    for line in lines:
        y, y_error, x, color = line
        axs[2].plot(x, y, color=color, marker='o')
        axs[2].fill_between(x, y - y_error, y + y_error, alpha=0.3, color=color)
    axs[2].plot(optimal_speedup, optimal_speedup, label='optimal scalability', color='#C1666B')
    axs[2].set_xlabel('number processes', fontsize=20)
    axs[2].set_xticks(optimal_speedup[1:], optimal_speedup[1:], fontsize=18)
    axs[2].set_title('(c) large synthetic', fontsize=23)

    handles, labels = axs[0].get_legend_handles_labels()
    line_1 = Line2D([0], [0], label='32 hash length', color='green')
    line_2 = Line2D([0], [0], label='128 hash length', color='blue')
    line_3 = Line2D([0], [0], label='512 hash length', color='orange')
    line_4 = Line2D([0], [0], label='2048 hash length', color='orchid')
    handles.extend([line_1, line_2, line_3, line_4])
    axs[0].legend(handles=handles, loc='best', fontsize=22)
    plt.savefig(f'paper_charts/speedup_pynndescent.png', bbox_inches='tight')


def course_dimensionality():
    qps_bruteforce_large = {
        'mean': [21.92516564147603, 20.576386547837263, 18.310599443821058, 14.294708699536937, 9.728681561762224, 5.839611474993033, 1.873244693686018, 0.4971154043701028, 0.12553415261935685],
        'error': [0.28196861759900654, 0.3708332160874859, 0.41090628667582585, 0.23541980405938187, 0.06614722074303771, 0.05301108559528854, 0.010735223090178669, 0.0064679272563895, 0.0027742345216329894]
    }
    qps_balltree_large = {
        'mean': [6185.71913051578, 4670.567601387252, 1646.5975292074813, 114.06120677200252, 7.54878799400503, 3.994141531252629, np.NaN, np.NaN, np.NaN],
        'error': [31.88189126987837, 5.969982718346613, 10.968007399278795, 0.2690578051671977, 0.023557347728966914, 0.016914164108496454, np.NaN, np.NaN, np.NaN]
    }
    qps_pynndescent_large = {
        'mean': [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, 11708.504448089661, 4042.2247382688497, 199.07459845906192, 2.704015756545677],
        'error': [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, 231.45319988448057, 80.30552987306814, 2.87626509918476, 0.055557669498237754]
    }
    qps_bruteforce_large_gpu = {
        'mean': [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, 21.104572589765752, 21.09609487083733, 19.734008523966025, 18.14725961274277],
        'error': [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, 0.6959737402052791, 0.15123938852234747, 1.2796876051427148, 0.3999890494492262]
    }
    colors = ['green', 'blue', 'orange', 'orchid']
    lines = [
        (
            np.asarray(qps_bruteforce_large['mean'])/1000,
            np.asarray(qps_bruteforce_large['error'])/1000,
            np.arange(0, len(qps_bruteforce_large['mean'])),
            colors[0]
        ),
        (
            np.asarray(qps_balltree_large['mean'])/1000,
            np.asarray(qps_balltree_large['error'])/1000,
            np.arange(0, len(qps_bruteforce_large['mean'])),
            colors[1]
        ),
        (
            np.asarray(qps_pynndescent_large['mean'])/1000,
            np.asarray(qps_pynndescent_large['error'])/1000,
            np.arange(0, len(qps_bruteforce_large['mean'])),
            colors[2]
        ),
        (
            np.asarray(qps_bruteforce_large_gpu['mean'])/1000,
            np.asarray(qps_bruteforce_large_gpu['error'])/1000,
            np.arange(0, len(qps_bruteforce_large['mean'])),
            colors[3]
        )

    ]

    plt.figure(figsize=(9, 5))
    ax = plt.gca()

    for line in lines:
        y, y_error, x, color = line
        plt.plot(x, y, color=color, marker='o')
        plt.fill_between(x, y - y_error, y + y_error, alpha=0.3, color=color)
    # plt.yticks([2, 4, 8, 16], [2, 4, 8, 16], fontsize=18)
    #plt.ylim(2, 25)
    plt.ylabel('Queries per second (1/s) in scale of 1000', fontsize=14)
    plt.xlabel('dimension size / hash lengths', fontsize=20)
    plt.yticks(fontsize=18)
    plt.xticks(np.arange(0, len(qps_bruteforce_large['mean'])), [1, 2, 4, 8, 16, 32, 128, 512, 2048], fontsize=18)

    handles, labels = ax.get_legend_handles_labels()
    line_1 = Line2D([0], [0], label='brute-force with CPU', color=colors[0])
    line_2 = Line2D([0], [0], label='BallTree', color=colors[1])
    line_3 = Line2D([0], [0], label='PyNNDescent', color=colors[2])
    line_4 = Line2D([0], [0], label='brute-force with GPU', color=colors[3])
    handles.extend([line_1, line_4, line_2, line_3])
    plt.legend(handles=handles, loc='upper left', fontsize=18)
    plt.savefig(f'paper_charts/course_of_dimensionality.png', bbox_inches='tight')


if __name__ == '__main__':
    speedup_data_vs_query_parallelism()
    speedup_cpu_vs_gpu()
    qps_cpu_vs_gpu()
    speedup_pynndescent()
    course_dimensionality()
