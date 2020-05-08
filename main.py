import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import scipy.stats as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

o_data_tasa = pd.read_csv("ten_nodes_tasa.csv", sep=";")
o_start_tasa = min(o_data_tasa.send.to_numpy())
o_end_tasa = (max(o_data_tasa.rec.to_numpy()) - o_start_tasa)
o_data_tasa.send -= o_start_tasa
o_data_tasa.rec -= o_start_tasa

o_data_msf = pd.read_csv("new_ten_msf.csv", sep=";")
o_start_msf = min(o_data_msf.send.to_numpy())
o_end_msf = (max(o_data_msf.rec.to_numpy()) - o_start_msf)
o_data_msf.send -= o_start_msf
o_data_msf.rec -= o_start_msf

o_end = min(o_end_tasa, o_end_msf)


def end_to_end_delay(data_msf, data_tasa, end, sampling_time=1000):
    history_msf = {}
    history_tasa = {}

    for i in range(0, end + 1):
        history_tasa[i] = 0
        history_msf[i] = 0

    for index, row in data_tasa.iterrows():
        send_time = row["send"] / sampling_time
        rec_time = row["rec"] / sampling_time
        # print "send at : {} -- rec at : {}".format(send_time, rec_time)
        for i in range(min(end, send_time), min(rec_time, end)):
            history_tasa[i] += 1

    for index, row in data_msf.iterrows():
        send_time = row["send"] / sampling_time
        rec_time = row["rec"] / sampling_time
        # print "send at : {} -- rec at : {}".format(send_time, rec_time)
        for i in range(min(end, send_time), min(rec_time, end)):
            history_msf[i] += 1
        # print "{} -- {}".format(row["send"], row["rec"])

    print "Min {}".format(0)
    print "duration {}".format(end / sampling_time)
    t = []
    msf = []
    tasa = []
    for key in history_msf:
        # print "At {} : msf -> {} ; tasa -> {}".format(key, history_msf[key], history_tasa[key])
        t.append(key)
        msf.append(history_msf[key])
        tasa.append(history_tasa[key])

    plt.plot(t, msf, marker="o", markersize=2, label="msf", color='r')
    plt.plot(t, tasa, marker='+', markersize=2, label="tasa", color="b")
    plt.xlabel('Time (s)')
    plt.ylabel('RRT packet count')

    plt.title("Number of COAP-RRT-data packets in network")
    plt.legend()

    plt.show()


def end_to_end_delay_empirical(data_msf, data_tasa):
    o_tasa_delays = data_tasa.rec.to_numpy() - data_tasa.send.to_numpy()
    o_msf_delays = data_msf.rec.to_numpy() - data_msf.send.to_numpy()
    labels = ["MSF", "TASA"]
    plt.ylim(top=1800)  # adjust the top leaving bottom unchanged
    plt.ylim(bottom=0)  # adjust the bottom leaving top unchanged
    data = [o_tasa_delays, o_msf_delays]
    plt.boxplot(data, showfliers=False)
    plt.xticks([1, 2], labels, rotation=60, fontsize=8)
    plt.title('End to end delay comparison')
    plt.ylabel("end-to-end delay (ms)")
    plt.xlabel("Algorithm")
    plt.legend()

    plt.show()
    print "Mean tasa : {} \n".format(np.median(o_tasa_delays))
    print "Mean msf : {} \n".format(np.median(o_msf_delays))


# ten nodes 0.994
def confidence_interval(data_msf, data_tasa, confidence=0.994):
    o_tasa_delays = (data_tasa.rec.to_numpy() - data_tasa.send.to_numpy()) * 1.0
    o_msf_delays = (data_msf.rec.to_numpy() - data_msf.send.to_numpy()) * 1.0
    msf_n, msf_mean, msf_std = len(o_msf_delays), np.mean(o_msf_delays), st.sem(o_msf_delays)
    tasa_n, tasa_mean, tasa_std = len(o_tasa_delays), np.mean(o_tasa_delays), st.sem(o_tasa_delays)

    msf_h = msf_std * st.t.ppf((1 + confidence) / 2, msf_n - 1)
    tasa_h = tasa_std * st.t.ppf((1 + confidence) / 2, tasa_n - 1)
    barWidth = 0.3

    data_mean = [msf_mean, tasa_mean]
    data_error = [msf_h, tasa_h]
    plt.bar(1, msf_mean, width=barWidth, color='blue', edgecolor='black', yerr=msf_h, capsize=3, label='MSF')
    plt.bar(2, tasa_mean, width=barWidth, color='red', edgecolor='black', yerr=tasa_h, capsize=3, label='TASA')

    plt.xticks([1, 2], ['MSF', 'TASA'])
    plt.title("End-to-end delay confidence interval (ms)")
    plt.ylabel("Mean of end-to-end delay")
    plt.legend()
    plt.show()
    print "TASA -- mean {} -- [{},{}]".format(tasa_mean, tasa_mean - tasa_h, tasa_mean + tasa_h)
    print "MSF  -- mean {} -- [{},{}]".format(msf_mean, msf_mean - msf_h, msf_mean + msf_h)


def queue_emptiness_by_time(data_msf, data_tasa, confidence=0.9999999):
    tasa_queue = data_tasa.queue.to_numpy()
    msf_queue = data_msf.queue.to_numpy()

    msf_n, msf_mean, msf_std = len(msf_queue), np.mean(msf_queue), st.sem(msf_queue)
    tasa_n, tasa_mean, tasa_std = len(tasa_queue), np.mean(tasa_queue), st.sem(tasa_queue)

    msf_h = msf_std * st.t.ppf((1 + confidence) / 2, msf_n - 1)
    tasa_h = tasa_std * st.t.ppf((1 + confidence) / 2, tasa_n - 1)

    barWidth = 0.3
    plt.bar(1, msf_mean, width=barWidth, color='blue', edgecolor='black', yerr=msf_h, capsize=2, label='MSF')

    # Create cyan bars
    plt.bar(2, tasa_mean, width=barWidth, color='red', edgecolor='black', yerr=tasa_h, capsize=2, label='TASA')
    plt.xticks([1, 2], ['MSF', 'TASA'])
    plt.title("Queue free level -- Confidence interval")
    plt.ylabel("Queue free level")
    plt.legend()
    plt.show()
    print "TASA-queue -- mean {} -- [{},{}]".format(tasa_mean, tasa_mean - tasa_h, tasa_mean + tasa_h)
    print "MSF_queue  -- mean {} -- [{},{}]".format(msf_mean, msf_mean - msf_h, msf_mean + msf_h)
    return


# end_to_end_delay(o_data_msf, o_data_tasa, o_end, sampling_time=10)

# end_to_end_delay_empirical(o_data_msf, o_data_tasa)

#confidence_interval(o_data_msf, o_data_tasa)
# end_to_end_delay_empirical(o_data_msf, o_data_tasa)

#queue_emptiness_by_time(o_data_msf, o_data_tasa)

# number_of_delivered_packets(o_data_msf, o_data_tasa, o_end)
#exit(1)

def exec_time():
    exec_data = pd.read_csv("exec_time.csv", sep=";")
    X = exec_data.X.to_numpy()
    Y = exec_data.Y.to_numpy()

    plt.plot(X, Y, 'o', color='black')
    X = np.array(X).reshape((-1, 1))
    Y = np.array(Y).reshape((-1, 1))
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.0, random_state=0)

    polynomial_features = PolynomialFeatures(degree=2)
    x_poly = polynomial_features.fit_transform(X)
    print x_poly
    model = LinearRegression()
    model.fit(x_poly, Y)
    y_poly_pred = model.predict(X)

    plt.show()


def get_total_number_of_packet(p_data):
    grouped_df = p_data.groupby("id_2")
    maximums = grouped_df.max()
    maximums = maximums.reset_index()
    return np.sum(maximums.packet_id.to_numpy())


import numpy as np
from matplotlib import pyplot as plt

def PolyCoefficients(x, coeffs):
    """ Returns a polynomial for ``x`` values for the ``coeffs`` provided.
    f(x) =  1.1906207901367749e-001 * x^0
         + -9.1103319111047951e-003 * x^1
         +  6.0626160696506065e-004 * x^2
    The coefficients must be in ascending order (``x**0`` to ``x**o``).
    """
    o = len(coeffs)
    #print(f'# This is a polynomial of order {ord}.')
    y = 0
    for i in range(o):
        y += coeffs[i]*x**i
    return y

x = np.linspace(0, 10, 11)
coeffs = [1.1906207901367749e-001, -9.1103319111047951e-003, 6.0626160696506065e-004 ]
y = np.array([np.sum(np.array([coeffs[i]*(j**i) for i in range(len(coeffs))])) for j in x])
plt.plot(x, y)
plt.show()

exit(1)
# exec_time()

packet_count_tasa = get_total_number_of_packet(o_data_tasa)

packet_count_msf = get_total_number_of_packet(o_data_msf)
END = (o_end / 1000) + 1
count_t = 0
count_m = 0
hist_t = [0] * (END + 100)
hist_m = [0] * (END + 100)
t = []
for i in range(0, END + 100):
    t.append(i)
for elm in o_data_tasa.rec.to_numpy():
    n_elm = elm / 1000
    if n_elm < END:
        print "T-Elm : {}".format(n_elm)
        count_t += 1
        for i in range(n_elm, END):
            hist_t[i] += 1

for elm in o_data_msf.rec.to_numpy():
    n_elm = elm / 1000
    if n_elm < END:
        print "M-Elm : {}".format(n_elm)
        count_m += 1
        for i in range(n_elm, END):
            hist_m[i] += 1
for i in range(END, END + 100):
    hist_m[i] = hist_m[END - 1]
    hist_t[i] = hist_t[END - 1]
print "Min send : {} -- Max Rec : {} -- count : {}".format(min(o_data_tasa.send.to_numpy()),
                                                           max(o_data_tasa.rec.to_numpy()), count_t)
print "Min send : {} -- Max Rec : {} -- count : {}".format(min(o_data_msf.send.to_numpy()),
                                                           max(o_data_msf.rec.to_numpy()), count_m)

Xt = (1.0 * np.array(hist_t)) / packet_count_tasa
Xm = (1.0 * np.array(hist_m)) / packet_count_msf

for i in Xt:
    print "XT : {}".format(i)

axes = plt.gca()
axes.set_xlim([0, 900])
axes.set_ylim([0.0, 1.0])

plt.plot(t, Xt, marker="o", markersize=4, label="Tasa", color='r')
plt.plot(t, Xm, marker="o", markersize=4, label="Msf", color='b')
plt.xlabel('Time (s)')
plt.ylabel('packet delivery rate %')
plt.title("Comparaison of packet delivery rate")
plt.legend()
plt.show()
exit(1)

# print "Slot MSF : {}".format((1.0 * (hist_m[780] - hist_m[20])) / (1.0 * (t[780] - t[20])))
# print "Slot TASA : {}".format((1.0 * (hist_t[780] - hist_t[20])) / (1.0 * (t[780] - t[20])))
t_ = np.array(t).reshape((-1, 1))
hist_m_ = np.array(hist_m).reshape((-1, 1))

hist_t_ = np.array(hist_t).reshape((-1, 1))
X_train, X_test, y_train, y_test = train_test_split(t_, hist_m_, test_size=0.2, random_state=0)
regressor = LinearRegression()
regressor.fit(X_train, y_train)  # training the algorithm
B_m = regressor.intercept_
A_m = regressor.coef_
msf_test_y = regressor.predict(X_test)
plt.plot(X_test, msf_test_y, marker="o", markersize=4, label="msf_pred")

X_train, X_test, y_train, y_test = train_test_split(t_, hist_t_, test_size=0.2, random_state=0)
regressor = LinearRegression()
regressor.fit(X_train, y_train)  # training the algorithm
tasa_test_y = regressor.predict(X_test)

plt.plot(X_test, tasa_test_y, marker="o", markersize=4, label="tasa_pred")

B_t = regressor.intercept_
A_t = regressor.coef_
print("TASA : {} -- {}".format(B_t, A_t))
print("MSF : {} -- {}".format(B_m, A_m))

plt.plot(t, hist_m, marker="o", markersize=4, label="msf", color='r')
plt.plot(t, hist_t, marker='+', markersize=4, label="tasa", color="b")
plt.xlabel('Time (s)')
plt.ylabel('Number of delivered packet')
plt.title("Number of delivered COAP-RRT-data packets by time")
plt.legend()

plt.show()
