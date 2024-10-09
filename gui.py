import os
import threading
import io

import customtkinter
from PIL import Image
from matplotlib import pyplot as plt

from genetic import GeneticAlgorithm

customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

g = None

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        # configure window
        self.title("CustomTkinter complex_example.py")
        self.geometry(f"{1100}x{580}")

        # configure grid layout (4x4)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)

        # create sidebar frame with widgets
        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="GenoPy", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        self.trainBtn = customtkinter.CTkButton(self.sidebar_frame, text="Train Algorithm", command=self.train_algorithm)
        self.trainBtn.grid(row=1, column=0, padx=20, pady=10)
        self.paramResetBtn = customtkinter.CTkButton(self.sidebar_frame, text="Reset Params to Default", command=self.train_algorithm)
        self.paramResetBtn.grid(row=2, column=0)
        self.refreshGraphBtn = customtkinter.CTkButton(self.sidebar_frame, text="Refresh graph", command=self.updateGraph)
        self.refreshGraphBtn.grid(row=3, column=0, padx=20, pady=10 )

        self.appearance_mode_label = customtkinter.CTkLabel(self.sidebar_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=5, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["Light", "Dark", "System"],
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=6, column=0, padx=20, pady=(10, 10))
        self.scaling_label = customtkinter.CTkLabel(self.sidebar_frame, text="UI Scaling:", anchor="w")
        self.scaling_label.grid(row=7, column=0, padx=20, pady=(10, 0))
        self.scaling_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["80%", "90%", "100%", "110%", "120%"],
                                                               command=self.change_scaling_event)
        self.scaling_optionemenu.grid(row=8, column=0, padx=20, pady=(10, 20))



        # create textbox
        self.textbox = customtkinter.CTkTextbox(self, width=250)
        self.textbox.grid(row=0, column=1, padx=(20, 0), pady=(20, 0), sticky="nsew")

        # create tabview
        self.tabview = customtkinter.CTkTabview(self, width=250)
        self.tabview.grid(row=0, column=2, padx=(20, 0), pady=(20, 0), sticky="nsew")
        self.tabview.add("Configuration")
        self.tabview.tab("Configuration").grid_columnconfigure(0, weight=1)  # configure grid of individual tabs

        self.selectionMethodOption = customtkinter.CTkOptionMenu(self.tabview.tab("Configuration"), dynamic_resizing=False,
                                                                 values=["Roulette", "Tournament"])
        self.selectionMethodOption.set("Roulette")
        self.selectionMethodOption.grid(row=0, column=0, padx=20, pady=(20, 10))


        self.mutationMethodOption = customtkinter.CTkOptionMenu(self.tabview.tab("Configuration"), dynamic_resizing=False,
                                                                values=["Swap", "Scramble"])
        self.mutationMethodOption.set("Swap")
        self.mutationMethodOption.grid(row=1, column=0, padx=20, pady=(20, 10))

        self.mutProb = customtkinter.CTkEntry(self.tabview.tab("Configuration"), placeholder_text="Mutation Probability")
        self.mutProb.grid(row=2, column=0, padx=20, pady=(20, 10))
        self.mutProb.delete(0, "end")
        self.mutProb.insert(0, "0.05")

        self.crossProb = customtkinter.CTkEntry(self.tabview.tab("Configuration"), placeholder_text="Crossover Probability")
        self.crossProb.grid(row=3, column=0, padx=20, pady=(20, 10))
        self.crossProb.delete(0, "end")
        self.crossProb.insert(0, "0.2")

        self.populationSize = customtkinter.CTkEntry(self.tabview.tab("Configuration"), placeholder_text="100")
        self.populationSize.grid(row=4, column=0, padx=20, pady=(20, 10))
        self.populationSize.delete(0, "end")
        self.populationSize.insert(0, "100")

        self.generations = customtkinter.CTkEntry(self.tabview.tab("Configuration"), placeholder_text="5")
        self.generations.grid(row=5, column=0, padx=20, pady=(20, 10))
        self.generations.delete(0, "end")
        self.generations.insert(0, "5")

        # self.graph = customtkinter.CTkImage(Image.open(os.path.join(os.getcwd(), "myplot.png")), size=(900,540))
        #
        # self.imgFrame = customtkinter.CTkLabel(self, text="", image=self.graph, width=250)
        # self.imgFrame.grid(row=1, columnspan=2,column=1, padx=20, pady=(20, 10))
        # set default values
        self.appearance_mode_optionemenu.set("Dark")
        self.scaling_optionemenu.set("100%")

        self.textbox.insert("0.0", "Output log for algorithm training")

    def open_input_dialog_event(self):
        dialog = customtkinter.CTkInputDialog(text="Type in a number:", title="CTkInputDialog")
        print("CTkInputDialog:", dialog.get_input())

    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        customtkinter.set_widget_scaling(new_scaling_float)

    def updateGraph(self):
        if g is None:
            return
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot the min, max, and average costs
        ax.plot(range(g.history["generations"]), g.history["mins"], label='Min Cost', marker='o')
        ax.plot(range(g.history["generations"]), g.history["maxes"], label='Max Cost', marker='x')
        ax.plot(range(g.history["generations"]), g.history["averages"], label='Average Cost', marker='s')

        # Add labels and title
        ax.set_xlabel('Generations')
        ax.set_ylabel('Cost')
        ax.set_title('Genetic Algorithm Cost over Generations')
        ax.legend()
        ax.grid(True)

        # Save the plot to a BytesIO buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)  # Close the figure to free up memory
        buf.seek(0)  # Move the cursor to the beginning of the BytesIO buffer

        # Use PIL to open the image from the buffer
        img = Image.open(buf)
        self.graph = customtkinter.CTkImage(img, size=(900, 540))
        self.imgFrame = customtkinter.CTkLabel(self, text="", image=self.graph, width=250)
        self.imgFrame.grid(row=1, columnspan=2, column=1, padx=20, pady=(20, 10))


    def train_algorithm(self):
        global g
        with open("Random100.tsp", "r") as f:
            for _ in range(7):
                f.readline()
            cities = []
            for i in range(100):
                _, x, y = f.readline().split()
                cities.append((float(x), float(y)))
        if g is None:
            g = GeneticAlgorithm(cities, int(self.populationSize.get()), self.selectionMethodOption.get().lower(), self.mutationMethodOption.get().lower(),
                                 float(self.mutProb.get()), float(self.crossProb.get()))
        gens = int(self.generations.get())
        trainThread = threading.Thread(target=g.run, args=(gens,))
        trainThread.start()
        self.textbox.delete("1.0", "end")
        currentGenerations = 0
        while True:
            if g.history["generations"] > currentGenerations:
                self.textbox.insert(f"end", "Generation: " + str(g.history["generations"]) + "\n")
                currentGenerations += 1
                self.updateGraph()
                if g.history["generations"] >= gens:
                    break





if __name__ == "__main__":
    app = App()
    app.mainloop()