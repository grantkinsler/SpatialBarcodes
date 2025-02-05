import scipy.spatial.distance as sci_dist
from PIL import Image
from scipy.spatial import ConvexHull
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.path import Path
import cv2
from matplotlib.colors import hsv_to_rgb
from skimage.measure import regionprops, label
Image.MAX_IMAGE_PIXELS = None
np.random.seed(0)
data_folder = '/Users/yaelheyman/RajLab Dropbox/Yael Heyman/SpatialBarcodes/ImagingData/2024-05-21_mouseexp_expression/projects/roi_2'
output_folder = data_folder + '/processedData/erosion_analysis'
cell_by_gene_path = data_folder + '/exports/cell_by_gene_matrix_20240606_10px_withbarcodes_atleast3.csv'
selected_mask = data_folder +  '/processedData/erosion_analysis/selected_mask.png'
transcripts_path = data_folder + '/exports/decode_20240604.csv'


class ErosionAnalysis:
    def __init__(self):
        self.load_data()
        self.mask = np.zeros((48115, 60616), dtype=np.uint8)
        self.MAX_DIM = 20000
        self.kernel_size = 25   # Increase this value to make each erosion more substantial
        self.scaling_factor = 1  # Default to no scaling
        self.erosion_iterations = 100
        self.pixel2nm = 107.11 # every pixel is 107.11 nm
    def load_data(self):
        self.cell_by_gene = pd.read_csv(cell_by_gene_path)
        self.original_centers = self.cell_by_gene[['center_x', 'center_y']].values.astype(float)
        self.centers = self.original_centers.copy()
        self.areas = self.cell_by_gene['area'].values
        self.cell_by_gene = pd.read_csv(cell_by_gene_path)

    def resize_image_and_coordinates(self, image):
        height, width = image.shape
        if max(height, width) > self.MAX_DIM:
            self.scaling_factor = self.MAX_DIM / max(height, width)
            image_pil = Image.fromarray(image)
            image_pil = image_pil.resize((int(width * self.scaling_factor), int(height * self.scaling_factor)), Image.NEAREST)
            self.centers *= self.scaling_factor
            return np.array(image_pil)
        self.scaling_factor = 1  # No scaling
        return image

    def check_mask(self):
        mask_file = output_folder +f'/num_iterations_{self.erosion_iterations}' + '/selected_mask.png'
        if os.path.exists(mask_file):
            print(f"Loading existing mask from {mask_file}")
            mask = Image.open(mask_file)
            mask = mask.convert('L')
            mask = np.array(mask)
            self.mask = self.resize_image_and_coordinates(mask)
        else:
            self.create_mask()

    def create_mask(self):
        fig, ax = plt.subplots(figsize=(10, 10))
        scatter = ax.scatter(self.centers[:, 0], self.centers[:, 1], s=15, c='red', alpha=0.5)
        plt.title('Select points to create a mask')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')

        def onselect(verts):
            path = Path(verts)
            self.mask.fill(0)
            cv2.fillPoly(self.mask, [np.array(verts, np.int32)], 255)

            self.mask = self.resize_image_and_coordinates(self.mask)
            mask_image = Image.fromarray(self.mask)
            mask_image.save(output_folder+f'/num_iterations_{self.erosion_iterations}'  + '/selected_mask.png')

            plt.figure(figsize=(10, 10))
            plt.imshow(self.mask, cmap='gray')
            plt.title('Initial Mask')
            plt.xlabel('X Coordinate')
            plt.ylabel('Y Coordinate')
            plt.show()

        polygon_selector = PolygonSelector(ax, onselect)
        plt.show()
    def calculate_ring_width(self, mask1, mask2):
        # Subtract mask2 from mask1 to get the ring
        ring_mask = mask1 & ~mask2
        labeled_ring = label(ring_mask)
        labeled_outer = label(mask1)
    
        # Find the properties of the labeled regions
        region_ring = regionprops(labeled_ring)[0]
        region_mask = regionprops(labeled_outer)[0]
    
        # Area and perimeter of the ring
        ring_area = region_ring.area
        ring_perimeter = region_mask.perimeter
    
        # Correct the area and perimeter back to original dimensions
        corrected_ring_area = ring_area / (self.scaling_factor ** 2)
        corrected_ring_perimeter = ring_perimeter / self.scaling_factor
    
        # Calculate the average width of the ring based on the corrected perimeter
        average_ring_width = corrected_ring_area / corrected_ring_perimeter
    
        # Convert to micrometers
        average_ring_width_um = average_ring_width * self.pixel2nm / 1000
    
        return average_ring_width_um
    def erode_and_save(self):
        
        eroded_coordinates = []
        ring_widths = []
        initial_mask = self.mask.copy()
        previous_mask = self.mask.copy()

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.kernel_size, self.kernel_size))

        for i in range(self.erosion_iterations):
            print(f"Erosion iteration {i + 1}")
            print(f"Sum of previous mask before erosion: {np.sum(previous_mask)}")
            self.mask = cv2.erode(self.mask, kernel)
            print(f"Sum of current mask after erosion: {np.sum(self.mask)}")

            difference_mask = cv2.subtract(previous_mask, self.mask)
            width = self.calculate_ring_width(previous_mask, self.mask)
            ring_widths.append({'Ring Width (um)': width })
            eroded_coords = []

            for j, (x, y) in enumerate(self.centers):
                if int(y) < difference_mask.shape[0] and int(x) < difference_mask.shape[1] and difference_mask[int(y), int(x)] > 0:
                    eroded_coords.append((j, x / self.scaling_factor, y / self.scaling_factor))
            
            eroded_coordinates.append(eroded_coords)
            if self.scaling_factor != 1:
                eroded_mask_image = Image.fromarray(cv2.resize(self.mask, (int(self.mask.shape[1] / self.scaling_factor), int(self.mask.shape[0] / self.scaling_factor)), interpolation=cv2.INTER_NEAREST))
            else:
                eroded_mask_image = Image.fromarray(self.mask)
            eroded_mask_image.save(output_folder+f'/num_iterations_{self.erosion_iterations}' + f'/eroded_mask_{i + 1}.png')
            eroded_df = pd.DataFrame(eroded_coords, columns=['Index', 'X', 'Y'])
            eroded_df.to_csv(output_folder+f'/num_iterations_{self.erosion_iterations}'  + f'/eroded_coords_{i + 1}_.csv', index=False)

            previous_mask = self.mask.copy()
            cells_in_ring = eroded_df['Index'].values
            self.cell_by_gene.loc[self.cell_by_gene['cell_id'].isin(cells_in_ring), 'Ring'] = i
        # Convert to DataFrame and save as CSV
        df_ring_widths = pd.DataFrame(ring_widths)
        output_csv_path = os.path.join(output_folder,f'num_iterations_{self.erosion_iterations}', 'ring_widths.csv')
        self.cell_by_gene.to_csv(os.path.join(output_folder,f'num_iterations_{self.erosion_iterations}', 'cell_by_gene_with_rings.csv'))
        df_ring_widths.to_csv(output_csv_path, index=False)
        # Generate dynamic colors for plotting
        hsv_colors = [(i / self.erosion_iterations, 1, 1) for i in range(self.erosion_iterations)]
        rgb_colors = hsv_to_rgb(hsv_colors)

        # Display the final eroded mask
        if self.scaling_factor != 1:
            initial_mask_resized = cv2.resize(initial_mask, (int(initial_mask.shape[1] / self.scaling_factor), int(initial_mask.shape[0] / self.scaling_factor)), interpolation=cv2.INTER_NEAREST)
        else:
            initial_mask_resized = initial_mask

        plt.figure(figsize=(10, 10))
        plt.imshow(initial_mask_resized, cmap='gray')
        for i, coords in enumerate(eroded_coordinates):
            if coords:
                _, x, y = zip(*coords)
                plt.scatter(x, y, s=10, c=[rgb_colors[i]], label=f'Iteration {i+1}', alpha=0.6)
        plt.title('Final Eroded Mask with Cell Locations')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.legend()
        plt.savefig(output_folder +f'/num_iterations_{self.erosion_iterations}' + '/final_mask.jpg')
        plt.show()


if __name__ == "__main__":
    analysis = ErosionAnalysis()
    analysis.check_mask()
    analysis.erode_and_save()
