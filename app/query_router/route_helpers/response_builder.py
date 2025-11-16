"""Response building functionality for the query router."""

from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

class ResponseBuilder:
    """Builds and formats responses based on query analysis."""
    
    def __init__(self, config):
        """Initialize with configuration."""
        self.config = config
        
    def build_response(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build a response based on query analysis.
        
        Args:
            analysis: The analysis results from QueryAnalyzer
            
        Returns:
            Dictionary containing routing parameters
        """
        # Start with default parameters
        params = self._get_default_parameters()
        
        # Update based on analysis
        self._update_from_analysis(params, analysis)
        
        # Apply any final adjustments
        self._apply_final_adjustments(params, analysis)
        
        return params
    
    def _get_default_parameters(self) -> Dict[str, Any]:
        """Get default routing parameters."""
        return {
            'top_k': 5,
            'rerank': False,
            'null_threshold': 0.5,
            'max_distance': 0.6,
            'doc_type': None,
            'technologies': [],
            'categories': [],
            'question_type': None,
            'confidence': 1.0,
            'needs_clarification': False,
            'clarification_options': None
        }
    
    def _update_from_analysis(self, params: Dict[str, Any], analysis: Dict[str, Any]) -> None:
        """Update parameters based on analysis results."""
        # Update with analysis results
        params.update({
            'technologies': analysis.get('technologies', []),
            'categories': analysis.get('categories', []),
            'question_type': analysis.get('question_type'),
            'confidence': analysis.get('confidence', 1.0),
            'needs_clarification': analysis.get('needs_clarification', False),
            'clarification_options': analysis.get('clarification_options')
        })
        
        # Handle certificate matches
        if analysis.get('certificates'):
            best_cert = max(analysis['certificates'], key=lambda x: x['confidence'])
            params.update({
                'doc_type': 'certificate',
                'cert_id': best_cert['id'],
                'cert_name': best_cert['name'],
                'confidence': best_cert['confidence']
            })
    
    def _apply_final_adjustments(self, params: Dict[str, Any], analysis: Dict[str, Any]) -> None:
        """Apply final adjustments to parameters."""
        # Handle structured summaries (broad but clear intent with multiple domains)
        if analysis.get('is_structured_summary'):
            num_domains = len(analysis.get('summary_domains', []))
            params.update({
                'top_k': 15,  # Increased from 10 for multi-domain coverage
                'rerank': True,
                'null_threshold': 0.55,
                'max_distance': 0.65,
                'is_structured_summary': True,
                'is_ambiguous': False,  # Override ambiguity detection
            })
            logger.info(f"Structured summary detected covering {num_domains} domains: {analysis.get('summary_domains')}")
            return

        # Adjust for broad questions
        if params['question_type'] == 'broad':
            params.update({
                'top_k': 10,
                'rerank': True,
                'null_threshold': 0.55,
                'max_distance': 0.65
            })

        # Adjust for specific questions
        elif params['question_type'] == 'specific':
            params.update({
                'top_k': 3,
                'rerank': True,
                'null_threshold': 0.4,
                'max_distance': 0.5
            })

        # Cap confidence at 1.0
        params['confidence'] = min(1.0, params.get('confidence', 1.0))
