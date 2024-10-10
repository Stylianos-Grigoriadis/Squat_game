
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.1, bottom=0.25)


scatter = ax.scatter(x_data[:1], y_data[:1], label='Scatter Points')
ax.set_xlim(-10, 110)
ax.set_ylim(-10, 110)


ax_slider = plt.axes([0.1, 0.1, 0.8, 0.05], facecolor='lightgoldenrodyellow')
slider = Slider(ax_slider, 'Points', 1, len(x_data), valinit=1, valstep=1)

# Update function for the slider
def update(val):
    num_points = int(slider.val)
    scatter.set_offsets(np.c_[x_data[:num_points], y_data[:num_points]])
    fig.canvas.draw_idle()


slider.on_changed(update)

ax.legend()
plt.show()
