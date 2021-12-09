import tools

# initializing verification tool

tool = tools.verification_tool('acquisition_files/*')

# checking ranges

tool.range_check()

# writing reports

tool.write_report()

# ploting time series

tool.plot_time_series()

# ploting FFTs

tool.plot_fft()