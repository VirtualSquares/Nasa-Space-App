from flask import Flask, render_template, request
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import uuid

app = Flask(__name__)

if not os.path.exists('static'):
    os.makedirs('static')

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/main', methods=["GET", "POST"])
def main():
    plot_url = None
    csv_url = None

    if request.method == "POST":
        if 'file' not in request.files:
            return "No file part", 400

        file = request.files['file']

        if file.filename == '':
            return "No selected file", 400

        # READ CSV DATA #
        df = pd.read_csv(file)

        x = df['rel_time(sec)']
        y = df['velocity(c/s)']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        ax1.plot(x, y, linewidth=2, linestyle="-", c="b", label="Original")
        ax1.set_title("Original Data (From CSV)")
        ax1.set_xlabel("Relative Time (sec)")
        ax1.set_ylabel("Velocity (c/s)")
        ax1.legend()

        window_length = 10
        polyorder = 2
        yy = savgol_filter(y, window_length=window_length, polyorder=polyorder)

        ax2.plot(x, yy, linewidth=2, linestyle="-", c="r", label="Filtered")
        ax2.set_title("Filtered Data")
        ax2.set_xlabel("Relative Time (sec)")
        ax2.set_ylabel("Velocity (c/s)")
        ax2.legend()

        plot_filename = f"plot_{uuid.uuid4()}.png"
        plot_filepath = os.path.join('static', plot_filename)
        plt.savefig(plot_filepath)
        plt.close()

        time_rel = x.to_numpy()
        velocity = yy

        # VARIABLES #
        final_list = []
        earthquake_times = []

        def slope(yStep, xStep):
            return abs(yStep / xStep)

        mean_velocity = np.mean(velocity)
        mask = velocity > mean_velocity
        possibilities = velocity[mask]
        corresponding_times = time_rel[mask]

        mSlope_possibilities = 0
        counter_possibilities = 0

        if len(possibilities) > 1:
            for vel in range(1, len(possibilities)):
                y_dif = possibilities[vel] - possibilities[vel - 1]
                x_dif = corresponding_times[vel] - corresponding_times[vel - 1]

                s = slope(y_dif, x_dif)

                counter_possibilities += 1
                mSlope_possibilities += s

            mSlope_possibilities /= counter_possibilities
        else:
            mSlope_possibilities = None

        last_peak = None
        last_peak_time = None

        for i in range(len(possibilities)):
            if possibilities[i] > mSlope_possibilities:
                last_peak = possibilities[i]
                last_peak_time = corresponding_times[i]
                break

        if last_peak is not None:
            final_list.append(last_peak)
            earthquake_times.append(last_peak_time)

            nearby_points = []
            range_threshold = 1.0
            for i in range(len(possibilities)):
                if abs(corresponding_times[i] - last_peak_time) <= range_threshold and len(final_list) < 4:
                    if possibilities[i] != last_peak:
                        nearby_points.append((possibilities[i], corresponding_times[i]))

            nearby_points.sort(key=lambda x: abs(x[1] - last_peak_time))

            for point in nearby_points:
                if len(final_list) < 4:
                    final_list.append(point[0])
                    earthquake_times.append(point[1])

        if final_list:
            # Create a DataFrame from the earthquake data
            earthquake_data = pd.DataFrame({
                'Earthquake Velocity (c/s)': final_list,
                'rel_time(sec)': earthquake_times
            })

            # Define the filename and path
            csv_filename = f"earthquake_data_{uuid.uuid4()}.csv"
            csv_filepath = os.path.join('static', csv_filename)

            # Save the DataFrame to a CSV file
            earthquake_data.to_csv(csv_filepath, index=False)

            # Prepare the CSV URL for rendering in the template
            csv_url = f"/static/{csv_filename}"

        plot_data(time_rel, velocity, earthquake_times, final_list)
        plot_url = plot_filename

    return render_template("main.html", plot_url=plot_url, csv_url=csv_url)

def plot_data(time, velocity, earthquake_times, earthquake):
    plt.figure(figsize=(10, 5))
    plt.plot(time, velocity, label='Velocity (c/s)', color='blue')
    plt.title('Velocity vs. Time (Earthquake Highlighted In Red)')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Velocity (c/s)')
    plt.grid()
    plt.legend()

    if len(earthquake_times) > 0:
        for eq_time in earthquake_times:
            plt.axvspan(eq_time - 200, eq_time + 200, color='red', alpha=0.9)
            plt.text(eq_time, max(velocity), f'EQ: {eq_time:.2f} sec', color='black',
                     ha='center', va='bottom', fontsize=9, bbox=dict(facecolor='white', alpha=0.5))

    plt.show()
    plt.close()

if __name__ == '__main__':
    app.run(debug=True)
