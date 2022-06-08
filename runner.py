from data_generator import DataGenerator
from data_visualizer import DataVisualizer

# Generate a dataframe extracting the various attributes from the file names
data_generator = DataGenerator()
data = data_generator.create_dataframe_from_filenames()

# Plot the distribution of the attributes
data_visualizer = DataVisualizer(data)
# data_visualizer.plot_data_distribution(['Emotions', 'Sex', 'Ethnicity', 'Race'])
# data_visualizer.plot_waveplot_spectrogram()