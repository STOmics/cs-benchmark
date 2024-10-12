import cv2
import numpy as np
import pandas as pd
import math
from skimage import measure
from scipy.spatial.distance import cdist

class CellMorphology:
    def __init__(self, contour_points, mask):
        """
        Initialize the CellMorphology class with the contour points of a cell.

        Args:
            contour_points (numpy.ndarray): Contour points of the cell.
        """
        self.contour_points = contour_points
        self.mask = mask

    def calculate_area(self):
        """
        Calculate the area of the cell using its contour.

        Returns:
            float: Area of the cell.
        """
        return cv2.contourArea(self.contour_points)

    def calculate_elongation(self):
        """
        Calculate the elongation (aspect ratio) of the cell.

        Returns:
            float: Elongation value, or None if the width is zero.
        """
        rect = cv2.minAreaRect(self.contour_points)
        box = cv2.boxPoints(rect)
        height = abs(box[1][1] - box[0][1])
        width = abs(box[1][0] - box[0][0])
        if width == 0:
            return None
        return height / width

    def calculate_compactness(self):
        """
        Calculate the compactness of the cell.

        Returns:
            float: Compactness value, or None if the perimeter is zero.
        """
        area = self.calculate_area()
        perimeter = cv2.arcLength(self.contour_points, True)
        if perimeter == 0:
            return None
        return (4 * math.pi * area) / (perimeter ** 2)

    def calculate_eccentricity(self):
        """
        Calculate the eccentricity of the cell.

        Returns:
            float: Eccentricity value, or None if the contour points are insufficient.
        """
        if len(self.contour_points) > 5:
            ellipse = cv2.fitEllipse(self.contour_points)
            axes = ellipse[1]
            width, height = axes[1] / 2, axes[0] / 2  # Width is the semi-major axis, height is the semi-minor axis
            if width != 0 and height != 0:
                return math.sqrt(1 - (height ** 2) / (width ** 2))
        return None

    def calculate_sphericity(self):
        """
        Calculate the sphericity of the cell.

        Returns:
            float: Sphericity value, or None if the contour points are insufficient.
        """
        if len(self.contour_points) >= 5:
            ellipse = cv2.fitEllipse(self.contour_points)
            major_axis = max(ellipse[1])
            minor_axis = min(ellipse[1])
            inscribed_radius = np.sqrt(major_axis * minor_axis) / 2
            enclosing_circle = cv2.minEnclosingCircle(self.contour_points)
            enclosing_radius = enclosing_circle[1]
            if enclosing_radius != 0:
                return inscribed_radius / enclosing_radius
        return None

    def calculate_convexity(self):
        """
        Calculate the convexity of the cell.

        Returns:
            float: Convexity value, or None if the perimeter is zero.
        """
        perimeter = cv2.arcLength(self.contour_points, True)
        if perimeter == 0:
            return None
        cell_convex_hull = cv2.convexHull(self.contour_points)
        convex_hull_perimeter = cv2.arcLength(cell_convex_hull, True)
        return convex_hull_perimeter / perimeter

    def calculate_solidity(self):
        """
        Calculate the solidity of the cell.

        Returns:
            float: Solidity value, or None if the convex hull area is zero.
        """
        area = self.calculate_area()
        cell_convex_hull = cv2.convexHull(self.contour_points)
        convex_hull_area = cv2.contourArea(cell_convex_hull)
        if convex_hull_area == 0:
            return None
        return area / convex_hull_area

    def calculate_circularity(self):
        """
        Calculate the circularity of the cell.

        Returns:
            float: Circularity value, or None if the area or convex hull perimeter is zero.
        """
        area = self.calculate_area()
        if area == 0:
            return None
        cell_convex_hull = cv2.convexHull(self.contour_points)
        convex_hull_perimeter = cv2.arcLength(cell_convex_hull, True)
        if convex_hull_perimeter == 0:
            return None
        return (4 * math.pi * area) / (convex_hull_perimeter ** 2)
    
    def cell_tightness(self):
        """
        Calculate the cell tightness and average distance between cells.

        Returns:
            tuple: Average distance between cells and tightness value.
        """
        if not self.contour_points.size:
            return None, None

        properties = measure.regionprops(measure.label(self.mask))
        if not properties:
            return None, None

        centroids = np.array([prop.centroid for prop in properties])
        areas = [prop.area for prop in properties]

        distances = cdist(centroids, centroids, 'euclidean')
        np.fill_diagonal(distances, np.inf)
        nearest_distances = np.min(distances, axis=1)

        average_distance = np.mean(nearest_distances)
        average_area = np.mean(areas) if areas else None

        if average_area is None or average_area == 0:
            tightness = None
        else:
            tightness = average_distance * average_distance / average_area

        return average_distance, tightness

class CellImageAnalyzer:
    def __init__(self, image_path):
        """
        Initialize the CellImageAnalyzer class with the image path.

        Args:
            image_path (str): The path to the image file.
        """
        self.image_path = image_path
        self.image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        self.stats = None
        self.morphology_stats = None

    def compute_contours_and_stats(self):
        """
        Compute contours and connected component statistics for the image.
        """
        _, labels, stats, _ = cv2.connectedComponentsWithStats(self.image)
        contours, _ = cv2.findContours(self.image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)

        new_stats = np.zeros((np.max(labels), 6), dtype=np.object_)
        for label in range(1, np.max(labels)):
            new_stats[label, :5] = stats[label, :5]
            new_stats[label, 5] = contours[label - 1]

        self.stats = new_stats

    def calculate_morphology_stats(self):
        """
        Calculate morphological statistics for each cell in the image.
        """
        if self.stats is None:
            raise ValueError("Please compute cell contours and statistics first.")

        self.morphology_stats = []
        for i in range(1, len(self.stats)):
            contour_points = np.array(self.stats[i][5])
            if contour_points.size == 0:
                self.morphology_stats.append([None] * 11)
                continue

            cell = CellMorphology(contour_points,self.image)

            cell_area = cell.calculate_area()
            cell_elongation = cell.calculate_elongation()
            cell_compactness = cell.calculate_compactness()
            cell_eccentricity = cell.calculate_eccentricity()
            cell_sphericity = cell.calculate_sphericity()
            cell_convexity = cell.calculate_convexity()
            cell_solidity = cell.calculate_solidity()
            cell_circularity = cell.calculate_circularity()
            average_distance, tightness = cell.cell_tightness()

            self.morphology_stats.append([
                cell_area, cell_elongation, cell_compactness, cell_eccentricity,
                cell_sphericity, cell_convexity, cell_solidity, cell_circularity,
                average_distance, tightness
            ])

    def generate_dataframe(self):
        """
        Generate a DataFrame containing morphological statistics.

        Returns:
            pandas.DataFrame: DataFrame with the calculated morphology statistics.
        """
        if self.morphology_stats is None:
            raise ValueError("Please calculate the morphological statistics first.")

        df = pd.DataFrame(self.morphology_stats, columns=[
            'cellArea', 'cellElongation', 'cellCompactness', 'cellEccentricity',
            'cellSphericity', 'cellConvexity', 'cellSolidity', 'cellCircularity',
            'averageDistance', 'cellTightness'
        ])

        df.index += 1
        df.index.name = 'Label'
        return df

# Example usage
# analyzer = CellImageAnalyzer("path/to/your/image.png")
# analyzer.compute_contours_and_stats()
# analyzer.calculate_morphology_stats()
# df = analyzer.generate_dataframe()
# print(df)
