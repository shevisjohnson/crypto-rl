import matplotlib.pyplot as plt
import numpy as np


class PortfolioGraph:
    """
    A stock trading visualization using matplotlib
    made to render OpenAI gym environments
    """
    plt.style.use('dark_background')

    def __init__(self, sym=None):
        # attributes for rendering
        self.sym = sym
        self.line1 = []
        self.screen_size = 500
        self.y_vec = None
        self.x_vec = np.linspace(0, self.screen_size * 10,
                                 self.screen_size + 1)[0:-1]
        self.trade_count = 0
        self.trade_count_text = None
        self.allocation = {}
        self.pie1 = None

    def reset_render_data(self, y_vec):
        self.y_vec = y_vec
        self.line1 = []
        self.pie1 = []

    def render(self, total_value=10000., trade_count=0, allocation={}, mode='human'):
        if mode == 'human':
            self.line1, self.pie1 = self.live_plotter(self.x_vec,
                                           self.y_vec,
                                           self.line1,
                                           self.pie1,
                                           self.trade_count,
                                           self.allocation,
                                           identifier=self.sym)
            self.trade_count = trade_count
            self.allocation = allocation
            self.y_vec = np.append(self.y_vec[1:], total_value)

    #@staticmethod
    def live_plotter(self, x_vec, y1_data, line1, pie1, trade_count, allocation, identifier='Add Symbol Name',
                     pause_time=0.00001):
        if not line1:
            # this is the call to matplotlib that allows dynamic plotting
            plt.ion()
            fig = plt.figure(figsize=(20, 12))
            ax = fig.add_subplot(111)
            #ax2 = fig.add_subplot(121)
            # create a variable for the line so we can later update it
            line1, = ax.plot(x_vec, y1_data, '-', label='Value', alpha=0.8)

            sizes = list(map(lambda x: 100.0 * x, allocation.values()))
            labels = list(allocation.keys())

            #pie1, _, _ = ax2.pie(sizes, labels=labels, autopct='%1.1f%%',
            #                shadow=True, startangle=90)

            # update plot label/title
            self.trade_count_text = ax.text(0.95, 0.01, f'Trade count: {trade_count}',
                verticalalignment='top', horizontalalignment='left',
                transform=ax.transAxes,
                color='green', fontsize=15
            )

            plt.ylabel('Value')
            plt.legend()
            plt.title('Title: {}'.format(identifier))
            plt.show(block=False)

        self.trade_count_text.set_text(f'Trade count: {trade_count}')

        #wedge_values = list(allocation.values())

        #cursor = 0.0

        #for i, wedge in enumerate(pie1):
        #    new_cursor = min((cursor + (wedge_values[i] * 360.0)), 360.0)
        #    wedge.set_theta1(cursor)
        #    wedge.set_theta1(new_cursor)
        #    cursor = new_cursor
        # after the figure, axis, and line are created, we only need to update the
        # y-data
        line1.set_ydata(y1_data)

        # adjust limits if new data goes beyond bounds
        if np.min(y1_data) <= line1.axes.get_ylim()[0] or \
                np.max(y1_data) >= line1.axes.get_ylim()[1]:
            plt.ylim(np.min(y1_data), np.max(y1_data))

        # this pauses the data so the figure/axis can catch up
        # - the amount of pause can be altered above
        plt.pause(pause_time)

        # return line so we can update it again in the next iteration
        return line1, pie1

    @staticmethod
    def close():
        plt.close()


class TradingGraph:
    """
    A portfolio visualization using matplotlib
    made to render OpenAI gym environments
    """
    plt.style.use('dark_background')

    def __init__(self, sym=None):
        # attributes for rendering
        self.sym = sym
        self.line1 = []
        self.screen_size = 500
        self.y_vec = None
        self.x_vec = np.linspace(0, self.screen_size * 10,
                                 self.screen_size + 1)[0:-1]

    def reset_render_data(self, y_vec):
        self.y_vec = y_vec
        self.line1 = []

    def render(self, midpoint=100., mode='human'):
        if mode == 'human':
            self.line1 = self.live_plotter(self.x_vec,
                                           self.y_vec,
                                           self.line1,
                                           identifier=self.sym)
            self.y_vec = np.append(self.y_vec[1:], midpoint)

    @staticmethod
    def live_plotter(x_vec, y1_data, line1, identifier='Add Symbol Name',
                     pause_time=0.00001):
        if not line1:
            # this is the call to matplotlib that allows dynamic plotting
            plt.ion()
            fig = plt.figure(figsize=(20, 12))
            ax = fig.add_subplot(111)
            # create a variable for the line so we can later update it
            line1, = ax.plot(x_vec, y1_data, '-', label='midpoint', alpha=0.8)
            # update plot label/title
            plt.ylabel('Price')
            plt.legend()
            plt.title('Title: {}'.format(identifier))
            plt.show(block=False)

        # after the figure, axis, and line are created, we only need to update the
        # y-data
        line1.set_ydata(y1_data)

        # adjust limits if new data goes beyond bounds
        if np.min(y1_data) <= line1.axes.get_ylim()[0] or \
                np.max(y1_data) >= line1.axes.get_ylim()[1]:
            plt.ylim(np.min(y1_data), np.max(y1_data))

        # this pauses the data so the figure/axis can catch up
        # - the amount of pause can be altered above
        plt.pause(pause_time)

        # return line so we can update it again in the next iteration
        return line1

    @staticmethod
    def close():
        plt.close()