from pdf2image import convert_from_path
import os
from PIL import Image
from app.config import Config
from app.logger_config import logger
import math

# Disable image size limits in PIL
Image.MAX_IMAGE_PIXELS = None

class ImageService:
    def __init__(self):
        """Initialize image processing service."""
        self.images_dir = os.path.join(Config.BASE_DIR, "images")
        if not os.path.exists(self.images_dir):
            os.makedirs(self.images_dir)
    
    def convert_pdf_pages(self, pdf_path, start_page=None, end_page=None):
        """
        Convert PDF pages to images maintaining original quality.
        Processes pages in batches for better performance.
        """
        try:
            total_pages = end_page - start_page + 1 if start_page and end_page else None
            logger.info(f"Starting PDF conversion of {total_pages} pages from {pdf_path}")
            
            # Process in batches of 5 pages for better performance
            BATCH_SIZE = 5
            image_paths = []
            current_start = start_page if start_page else 1
            
            while current_start <= (end_page if end_page else float('inf')):
                current_end = min(current_start + BATCH_SIZE - 1, end_page if end_page else float('inf'))
                logger.info(f"Converting batch: pages {current_start}-{current_end} of {total_pages}")
                
                # Convert batch of pages with higher DPI for better quality
                batch_images = convert_from_path(
                    pdf_path,
                    first_page=current_start,
                    last_page=current_end,
                    dpi=200,  # Increased DPI for better quality
                    thread_count=5
                )
                
                # Process and save each image in the batch
                for i, image in enumerate(batch_images):
                    page_num = current_start + i
                    image_path = os.path.join(self.images_dir, f'page_{page_num}.jpg')
                    
                    # Save image with maximum quality
                    image.save(image_path, 'JPEG', quality=90)
                    image_paths.append(image_path)
                    
                    logger.info(f"Converted page {page_num}/{total_pages} to {image_path}")
                
                logger.info(f"Completed batch {current_start}-{current_end}")
                current_start += BATCH_SIZE
                
                if not end_page and not batch_images:  # No more pages to process
                    break
            
            logger.info(f"PDF conversion complete. Total pages converted: {len(image_paths)}")
            return image_paths
        
        except Exception as e:
            logger.error(f"Error during PDF conversion: {str(e)}")
            raise