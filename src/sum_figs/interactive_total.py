from typing import cast
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from abc import ABC, abstractmethod

from pathlib import Path

from PySide6 import QtWidgets, QtCore, QtGui
import sys

from icecream import ic

COLORMAPS = [
    "viridis",
    "plasma",
    "inferno",
    "magma",
    "cividis",
    "Greys",
    "Purples",
    "Blues",
    "Greens",
    "Oranges",
    "Reds",
    "turbo",
    "jet",
    "hot",
    "cool",
]


class Option(ABC):
    def __init__(self, name, parent: QtWidgets.QWidget):
        self.name = name
        self.parent = parent

    @abstractmethod
    def get_parts(self) -> tuple[QtWidgets.QWidget, ...]:
        pass

    @abstractmethod
    def value(self):
        pass

    @abstractmethod
    def add_to_grid(
        self, grid: QtWidgets.QGridLayout, row: int, column: int
    ) -> tuple[int, int]:
        pass


class FolderOption(Option):
    def __init__(self, name, parent: QtWidgets.QWidget):
        super().__init__(name, parent)

        self.label = QtWidgets.QLabel(f"{self.name}:")

        self.entry = QtWidgets.QLineEdit()
        self.entry.setReadOnly(True)

        self.button = QtWidgets.QPushButton("Browse...")

        self.button.clicked.connect(self._browse)
        self.selected = False

    def get_parts(self) -> tuple[QtWidgets.QWidget, ...]:
        return self.label, self.entry, self.button

    def value(self):
        return Path(self.entry.text()) if self.selected else None

    def add_to_grid(
        self, grid: QtWidgets.QGridLayout, row: int, column: int
    ) -> tuple[int, int]:
        grid.addWidget(self.label, row, column)
        grid.addWidget(self.entry, row + 1, column, 1, 2)
        grid.addWidget(self.button, row + 1, column + 2, 1, 1)
        return (2, 3)

    def _browse(self):
        dialog = QtWidgets.QFileDialog(self.parent, self.name)
        dialog.setFileMode(QtWidgets.QFileDialog.Directory)
        dialog.setOptions(QtWidgets.QFileDialog.ShowDirsOnly)

        if dialog.exec():
            selected = dialog.selectedFiles()[0]
            self.entry.setText(selected)
            self.selected = True

        self.parent.activateWindow()
        self.parent.raise_()


class ChoicesOption(Option):
    def __init__(self, name, choices: list, parent: QtWidgets.QWidget):
        super().__init__(name, parent)

        self.choices = choices

        self.label = QtWidgets.QLabel(f"{self.name}:")

        self.combo_box = QtWidgets.QComboBox()
        self.combo_box.addItems(choices)
        self.combo_box.setCurrentIndex(0)

    def get_parts(self) -> tuple[QtWidgets.QWidget, ...]:
        return self.label, self.combo_box

    def value(self):
        return self.choices[self.combo_box.currentIndex()]

    def add_to_grid(
        self, grid: QtWidgets.QGridLayout, row: int, column: int
    ) -> tuple[int, int]:
        grid.addWidget(self.label, row, column)
        grid.addWidget(self.combo_box, row, column + 1)
        return (1, 2)


class BoolOption(Option):
    def __init__(self, name, default: bool, parent: QtWidgets.QWidget):
        super().__init__(name, parent)

        self.check_box = QtWidgets.QCheckBox(f"{self.name}")
        self.check_box.setChecked(default)

    def get_parts(self) -> tuple[QtWidgets.QWidget, ...]:
        return (self.check_box,)

    def value(self):
        return self.check_box.isChecked()

    def add_to_grid(
        self, grid: QtWidgets.QGridLayout, row: int, column: int
    ) -> tuple[int, int]:
        grid.addWidget(self.check_box, row, column, 1, 2)
        return (1, 2)


def add_validator_for_type(typ: type, entry: QtWidgets.QLineEdit):
    if typ == float:
        validator = QtGui.QDoubleValidator(-1e12, 1e12, 12, entry)
        validator.setNotation(QtGui.QDoubleValidator.ScientificNotation)
        entry.setValidator(validator)
    elif typ == int:
        validator = QtGui.QIntValidator(entry)
        entry.setValidator(validator)
    elif typ == str:
        pass
    else:
        raise ValueError(f"Type {typ} not supported")


class InputOption(Option):
    def __init__(self, name, default: int | float | str, parent: QtWidgets.QWidget):
        super().__init__(name, parent)

        self.label = QtWidgets.QLabel(f"{self.name}:")
        self.entry = QtWidgets.QLineEdit(str(default))
        self.default = default

        add_validator_for_type(type(default), self.entry)

    def get_parts(self) -> tuple[QtWidgets.QWidget, ...]:
        return (self.label, self.entry)

    def value(self):
        if isinstance(self.default, float):
            return float(self.entry.text().strip())
        elif isinstance(self.default, int):
            return int(self.entry.text().strip())
        else:
            return self.entry.text().strip()

    def add_to_grid(
        self, grid: QtWidgets.QGridLayout, row: int, column: int
    ) -> tuple[int, int]:
        grid.addWidget(self.label, row, column, 1, 1)
        grid.addWidget(self.entry, row, column + 1, 1, 1)
        return (1, 2)


class ItemWidget(QtWidgets.QWidget):
    def __init__(self, values: list, remove_callback):
        super().__init__()
        self.values = values
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(4, 2, 4, 2)
        self.labels = [QtWidgets.QLabel(text) for text in self.values]
        for label in self.labels:
            label.setSizePolicy(
                QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred
            )
            layout.addWidget(label)
        self.del_btn = QtWidgets.QPushButton("Delete")
        self.del_btn.setFixedWidth(70)
        layout.addWidget(self.del_btn)
        self.setLayout(layout)
        self.del_btn.clicked.connect(remove_callback)


class ItemsOption(Option):
    def __init__(
        self, name, types: list[type], expected_height: int, parent: QtWidgets.QWidget
    ):
        super().__init__(name, parent)

        self.types = types
        self.expected_height = expected_height
        self.input_layout = QtWidgets.QHBoxLayout()
        self.entries = [QtWidgets.QLineEdit() for v in self.types]
        for entry, t in zip(self.entries, self.types):
            entry.setPlaceholderText("Type an item and press + or Enter")
            add_validator_for_type(t, entry)
            self.input_layout.addWidget(entry)
        self.add_button = QtWidgets.QPushButton("+")
        self.add_button.setFixedWidth(30)
        self.input_layout.addWidget(self.add_button)

        self.list_widget = QtWidgets.QListWidget()

        self.add_button.clicked.connect(self._add_item_from_entries)

    def _add_item_from_entries(self):
        values = [entry.text().strip() for entry in self.entries]
        self.add_item(values, raise_on_invalid=False)

    def add_item(self, values: list, raise_on_invalid: bool = True):
        assert len(values) == len(self.types)
        valid = True
        for entry, value in zip(self.entries, values):
            if validator := entry.validator():
                valid &= (
                    validator.validate(value, 0)[0] == QtGui.QValidator.State.Acceptable
                )
        if not valid:
            if raise_on_invalid:
                raise ValueError("One of the values provided was invalid.")
            return
        item = QtWidgets.QListWidgetItem()

        def remove():
            row = self.list_widget.row(item)
            if row != -1:
                self.list_widget.takeItem(row)

        widget = ItemWidget(values, remove)
        item.setSizeHint(widget.sizeHint())
        self.list_widget.addItem(item)
        self.list_widget.setItemWidget(item, widget)
        for entry in self.entries:
            entry.clear()
        self.entries[0].setFocus()

    def get_parts(self):
        return (
            self.input_layout,
            self.list_widget,
        )

    def value(self):
        values = []
        for ii in range(self.list_widget.count()):
            item = self.list_widget.item(ii)
            widget = self.list_widget.itemWidget(item)
            values.append(
                [t(v) for t, v in zip(self.types, cast(ItemWidget, widget).values)]
            )

        return values

    def add_to_grid(
        self, grid: QtWidgets.QGridLayout, row: int, column: int
    ) -> tuple[int, int]:
        grid.addLayout(self.input_layout, row, column, 1, len(self.types))
        grid.addWidget(
            self.list_widget, row + 1, column, self.expected_height, len(self.types)
        )
        return (self.expected_height + 1, len(self.types))


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Select In/Out Directories and Colormap")
        self.setMinimumSize(600, 140)
        self._create_widgets()
        self._layout_widgets()

    def _create_widgets(self):
        self.folder_in = FolderOption("Input Directory", self)
        self.folder_out = FolderOption("Input Directory", self)
        self.colormap = ChoicesOption("Colormap", COLORMAPS, self)
        self.process = BoolOption("Process", False, self)
        self.flt = InputOption("Float", 12.0, self)
        self.int = InputOption("Int", 12, self)
        self.ranges = ItemsOption("Ranges", [float, float], 4, self)

        self.btn_ok = QtWidgets.QPushButton("OK")
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_ok.clicked.connect(self.on_ok)
        self.btn_cancel.clicked.connect(self.close)

    def _layout_widgets(self):
        grid = QtWidgets.QGridLayout(self)
        grid.setContentsMargins(12, 12, 12, 12)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)

        row = 0
        height, _ = self.folder_in.add_to_grid(grid, row, 0)
        row += height
        height, _ = self.folder_out.add_to_grid(grid, row, 0)
        row += height
        height, _ = self.colormap.add_to_grid(grid, row, 0)
        row += height
        height, _ = self.process.add_to_grid(grid, row, 0)
        row += height
        height, _ = self.flt.add_to_grid(grid, row, 0)
        row += height
        height, _ = self.int.add_to_grid(grid, row, 0)
        row += height
        height, _ = self.ranges.add_to_grid(grid, row, 0)
        row += height

        btn_hbox = QtWidgets.QHBoxLayout()
        btn_hbox.addStretch()
        btn_hbox.addWidget(self.btn_ok)
        btn_hbox.addWidget(self.btn_cancel)
        grid.addLayout(btn_hbox, row, 0, 1, 3)

    def on_ok(self):
        in_dir = self.folder_in.value()
        out_dir = self.folder_out.value()
        cmap = self.colormap.value()
        process = self.process.value()
        flt = self.flt.value()
        int_value = self.int.value()
        ranges = self.ranges.value()

        if not in_dir or not out_dir:
            QtWidgets.QMessageBox.warning(
                self, "Missing", "Please select both input and output directories."
            )
            return

        print("Input directory:", in_dir)
        print("Output directory:", out_dir)
        print("Colormap:", cmap)
        print("Process:", process)
        print("Float:", flt)
        print("Int:", int_value)
        print("Ranges:", ranges)
        self.close()


def launch_chooser():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    # Try to ensure window gets focus
    win.show()
    win.raise_()
    win.activateWindow()
    # On some platforms you may prefer native dialogs: remove DontUseNativeDialog in browse_* above
    sys.exit(app.exec())


def show_interactively(
    data: np.ndarray,
    colormap,
    vmin0: float,
    vmax0: float,
    extent: tuple[float, float, float, float],
    origin: str,
):

    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.12, bottom=0.25)

    im = ax.imshow(
        data, cmap=colormap, vmin=vmin0, vmax=vmax0, extent=extent, origin=origin
    )
    fig.colorbar(im, ax=ax)

    # slider axes
    ax_vmin = plt.axes((0.12, 0.12, 0.76, 0.03))
    ax_vmax = plt.axes((0.12, 0.06, 0.76, 0.03))

    # slider ranges chosen from data min/max
    data_min, data_max = float(np.nanmin(data)), float(np.nanmax(data))
    slider_vmin = Slider(ax_vmin, "vmin", data_min, data_max, valinit=vmin0)
    slider_vmax = Slider(ax_vmax, "vmax", data_min, data_max, valinit=vmax0)

    def update(val):
        vmin = slider_vmin.val
        vmax = slider_vmax.val
        # enforce vmin <= vmax
        if vmin > vmax:
            # adjust the other slider to keep ordering
            if val is slider_vmin.val:
                slider_vmax.set_val(vmin)
                vmax = vmin
            else:
                slider_vmin.set_val(vmax)
                vmin = vmax
        im.set_clim(vmin, vmax)
        fig.canvas.draw_idle()

    slider_vmin.on_changed(update)
    slider_vmax.on_changed(update)

    # reset button
    reset_ax = plt.axes((0.12, 0.020, 0.08, 0.04))
    reset_btn = Button(reset_ax, "Reset", hovercolor="0.975")

    save_ax = plt.axes((0.80, 0.020, 0.08, 0.04))
    save_btn = Button(save_ax, "save", hovercolor="0.975")

    def reset(event):
        slider_vmin.reset()
        slider_vmax.reset()

    reset_btn.on_clicked(reset)

    def close(event):
        plt.close(fig)

    save_btn.on_clicked(close)

    plt.show()

    return slider_vmin.val, slider_vmax.val
