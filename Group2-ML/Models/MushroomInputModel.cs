using System.ComponentModel.DataAnnotations;

namespace Group2_ML.Models
{
    public class MushroomInputModel
    {
        [Required]
        public string CapShape { get; set; }

        [Required]
        public string CapSurface { get; set; }

        [Required]
        public string CapColor { get; set; }

        [Required]
        public string Bruises { get; set; }

        [Required]
        public string Odor { get; set; }

        [Required]
        public string GillAttachment { get; set; }

        [Required]
        public string GillSpacing { get; set; }

        [Required]
        public string GillSize { get; set; }

        [Required]
        public string GillColor { get; set; }

        [Required]
        public string StalkShape { get; set; }

        [Required]
        public string StalkRoot { get; set; }

        [Required]
        public string StalkSurfaceAboveRing { get; set; }

        [Required]
        public string StalkSurfaceBelowRing { get; set; }

        [Required]
        public string StalkColorAboveRing { get; set; }

        [Required]
        public string StalkColorBelowRing { get; set; }

        [Required]
        public string VeilType { get; set; }

        [Required]
        public string VeilColor { get; set; }

        [Required]
        public string RingNumber { get; set; }

        [Required]
        public string RingType { get; set; }

        [Required]
        public string SporePrintColor { get; set; }

        [Required]
        public string Population { get; set; }

        [Required]
        public string Habitat { get; set; }
    }
}
