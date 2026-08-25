"""End-to-end test: One file per datatype from Wizama project."""

import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
_logger = logging.getLogger(__name__)

PROJECT_DIR = Path("/media/sparrows/KINGSTON/Wizama - Social Game Console")

# Import our modules
sys.path.insert(0, str(Path(__file__).parent))
from acquire.metadata import ImageOCR, SensitiveDocFlagger, FilenamePatternDetector, CADMetadataExtractor
from clean.cleaner import (
    TextCleaner, PDFCleaner, ImageCleaner,
    XLSXCleaner, DOCXCleaner, FileCleanerRouter
)
from clean.anonymizer import EntityMapper

def find_one_file(ext: str) -> Path:
    """Find one file with given extension."""
    for f in PROJECT_DIR.rglob(f'*{ext}'):
        if f.is_file() and not f.name.startswith('._'):
            return f
    return None

def test_cleaner(name, cleaner, input_file, output_dir):
    """Test a cleaner on a file."""
    output_file = output_dir / input_file.name
    _logger.info(f"Testing {name} on {input_file.name}")
    try:
        result = cleaner.clean_file(input_file, output_file)
        size_in = input_file.stat().st_size
        size_out = output_file.stat().st_size if output_file.exists() else 0
        _logger.info(f"  OK: {size_in} -> {size_out} bytes")
        return True
    except Exception as e:
        _logger.error(f"  FAILED: {e}")
        return False

def main():
    output_dir = Path("/tmp/project-p-e2e-test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    mapper = EntityMapper()
    results = {}
    
    # Test 1: PDF
    pdf_file = find_one_file('.pdf')
    if pdf_file:
        cleaner = PDFCleaner(mapper)
        results['PDF'] = test_cleaner('PDFCleaner', cleaner, pdf_file, output_dir)
    
    # Test 2: JPG image
    jpg_file = find_one_file('.jpg') or find_one_file('.JPG')
    if jpg_file:
        cleaner = ImageCleaner(mapper)
        results['JPG'] = test_cleaner('ImageCleaner', cleaner, jpg_file, output_dir)
    
    # Test 3: PNG image
    png_file = find_one_file('.png')
    if png_file:
        cleaner = ImageCleaner(mapper)
        results['PNG'] = test_cleaner('ImageCleaner', cleaner, png_file, output_dir)
    
    # Test 4: STEP CAD file (metadata extraction only)
    step_file = find_one_file('.step') or find_one_file('.STEP')
    if step_file:
        extractor = CADMetadataExtractor()
        _logger.info(f"Testing CADMetadataExtractor on {step_file.name}")
        try:
            meta = extractor.extract_metadata(step_file)
            _logger.info(f"  OK: {len(meta)} metadata fields extracted")
            results['STEP'] = True
        except Exception as e:
            _logger.error(f"  FAILED: {e}")
            results['STEP'] = False
    
    # Test 5: Eagle SCH file (metadata extraction)
    sch_file = find_one_file('.sch')
    if sch_file:
        extractor = CADMetadataExtractor()
        _logger.info(f"Testing CADMetadataExtractor on {sch_file.name}")
        try:
            meta = extractor.extract_metadata(sch_file)
            _logger.info(f"  OK: {len(meta)} metadata fields extracted")
            results['SCH'] = True
        except Exception as e:
            _logger.error(f"  FAILED: {e}")
            results['SCH'] = False
    
    # Test 6: Eagle BRD file (metadata extraction)
    brd_file = find_one_file('.brd')
    if brd_file:
        extractor = CADMetadataExtractor()
        _logger.info(f"Testing CADMetadataExtractor on {brd_file.name}")
        try:
            meta = extractor.extract_metadata(brd_file)
            _logger.info(f"  OK: {len(meta)} metadata fields extracted")
            results['BRD'] = True
        except Exception as e:
            _logger.error(f"  FAILED: {e}")
            results['BRD'] = False
    
    # Test 7: XLSX
    xlsx_file = find_one_file('.xlsx')
    if xlsx_file:
        cleaner = XLSXCleaner(mapper)
        results['XLSX'] = test_cleaner('XLSXCleaner', cleaner, xlsx_file, output_dir)
    
    # Test 8: XLSM
    xlsm_file = find_one_file('.xlsm')
    if xlsm_file:
        cleaner = XLSXCleaner(mapper)
        results['XLSM'] = test_cleaner('XLSXCleaner', cleaner, xlsm_file, output_dir)
    
    # Test 9: XLS (legacy - will copy as-is, no cleaner)
    xls_file = find_one_file('.xls')
    if xls_file:
        router = FileCleanerRouter(mapper)
        results['XLS'] = test_cleaner('FileCleanerRouter', router, xls_file, output_dir)
    
    # Test 10: DOCX
    docx_file = find_one_file('.docx')
    if docx_file:
        cleaner = DOCXCleaner(mapper)
        results['DOCX'] = test_cleaner('DOCXCleaner', cleaner, docx_file, output_dir)
    
    # Test 11: DOC (legacy - will copy as-is)
    doc_file = find_one_file('.doc')
    if doc_file:
        router = FileCleanerRouter(mapper)
        results['DOC'] = test_cleaner('FileCleanerRouter', router, doc_file, output_dir)
    
    # Test 12: TXT (text cleaner)
    txt_file = find_one_file('.txt')
    if txt_file:
        cleaner = TextCleaner(mapper)
        results['TXT'] = test_cleaner('TextCleaner', cleaner, txt_file, output_dir)
    
    # Test 13: STL (3D print - copy as-is)
    stl_file = find_one_file('.stl')
    if stl_file:
        router = FileCleanerRouter(mapper)
        results['STL'] = test_cleaner('FileCleanerRouter', router, stl_file, output_dir)
    
    # Test 14: SensitiveDocFlagger on filenames
    _logger.info("Testing SensitiveDocFlagger on sample filenames")
    flagger = SensitiveDocFlagger()
    test_names = [
        "Quotation_2020-07-21-SquareOne Game Console.pdf",
        "FactoryAssemblyTest.jpg",
        "USB C spec.jpg",
        "box - SquareOne REV5.step",
    ]
    flagged = flagger.flag_files(test_names)
    _logger.info(f"  Flagged {len(flagged)}/{len(test_names)} as sensitive")
    results['SensitiveDocFlagger'] = True
    
    # Test 15: FilenamePatternDetector
    _logger.info("Testing FilenamePatternDetector on sample filenames")
    detector = FilenamePatternDetector()
    for name in test_names:
        patterns = detector.detect_patterns(name)
        if patterns:
            _logger.info(f"  {name}: {len(patterns)} patterns")
    results['FilenamePatternDetector'] = True
    
    # Summary
    _logger.info("")
    _logger.info("=" * 60)
    _logger.info("E2E TEST SUMMARY")
    _logger.info("=" * 60)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    for name, ok in sorted(results.items()):
        status = "PASS" if ok else "FAIL"
        _logger.info(f"  {name:25s} {status}")
    _logger.info(f"Total: {passed}/{total} passed")

if __name__ == '__main__':
    main()