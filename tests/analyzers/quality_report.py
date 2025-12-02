"""
Quality Report Generator - Generate comprehensive analysis reports.

Produces JSON, Markdown, and console output formats.
"""

from typing import Dict, Any, List
from datetime import datetime
from collections import defaultdict


class QualityReportGenerator:
    """
    Generates quality analysis reports in multiple formats.
    """

    def generate_report(self, analysis_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate comprehensive quality report.

        Args:
            analysis_results: List of analysis dicts from TestQualityAnalyzer

        Returns:
            Report dict with summary, breakdowns, and recommendations
        """
        return {
            'timestamp': datetime.now().isoformat(),
            'summary': self._generate_summary(analysis_results),
            'false_positives': self._identify_false_positives(analysis_results),
            'issue_breakdown': self._breakdown_by_type(analysis_results),
            'category_analysis': self._analyze_by_category(analysis_results),
            'severity_distribution': self._severity_distribution(analysis_results),
            'critical_findings': self._extract_critical(analysis_results),
            'recommendations': self._generate_recommendations(analysis_results),
            'test_details': analysis_results
        }

    def _generate_summary(self, analysis_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary metrics."""
        total_tests = len(analysis_results)
        tests_with_issues = sum(1 for r in analysis_results if r['issues'])
        total_issues = sum(len(r['issues']) for r in analysis_results)

        passed_tests = sum(1 for r in analysis_results if r['test_result']['passed'])
        passed_with_issues = sum(1 for r in analysis_results
                                 if r['test_result']['passed'] and r['issues'])

        # Count by issue type
        issue_type_counts = {}
        for result in analysis_results:
            for issue in result['issues']:
                issue_type = issue['type']
                issue_type_counts[issue_type] = issue_type_counts.get(issue_type, 0) + 1

        return {
            'total_tests': total_tests,
            'tests_with_quality_issues': tests_with_issues,
            'total_quality_issues': total_issues,
            'passed_tests': passed_tests,
            'false_positives': passed_with_issues,
            'false_positive_rate': (passed_with_issues / passed_tests * 100) if passed_tests > 0 else 0,
            'issue_type_counts': issue_type_counts
        }

    def _identify_false_positives(self, analysis_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Identify tests that passed but have quality issues (false positives).
        Priority detection for user.
        """
        false_positives = []

        for result in analysis_results:
            if result['test_result']['passed'] and result['issues']:
                # Calculate severity score for sorting
                severity_score = 0
                for issue in result['issues']:
                    severity = issue.get('severity', 'LOW')
                    severity_score += {'CRITICAL': 1000, 'HIGH': 100, 'MEDIUM': 10, 'LOW': 1}.get(severity, 0)

                false_positives.append({
                    'test_id': result['test_id'],
                    'category': result['category'],
                    'score': result['test_result']['validation_score'],
                    'question': result['question'],
                    'answer_preview': result['answer'][:100] + '...' if len(result['answer']) > 100 else result['answer'],
                    'issues': [
                        {
                            'type': issue['type'],
                            'severity': issue.get('severity'),
                            'description': issue['description'],
                            'impact': issue.get('impact')
                        }
                        for issue in result['issues']
                    ],
                    'issue_count': len(result['issues']),
                    'severity_score': severity_score
                })

        # Sort by severity score (highest first)
        false_positives.sort(key=lambda x: x['severity_score'], reverse=True)

        return false_positives

    def _breakdown_by_type(self, analysis_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Group issues by type with examples."""
        by_type = {}

        for result in analysis_results:
            for issue in result['issues']:
                issue_type = issue['type']
                if issue_type not in by_type:
                    by_type[issue_type] = {
                        'count': 0,
                        'severity_distribution': {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0},
                        'examples': []
                    }

                by_type[issue_type]['count'] += 1
                severity = issue.get('severity', 'MEDIUM')
                by_type[issue_type]['severity_distribution'][severity] += 1

                # Keep up to 3 examples per issue type
                if len(by_type[issue_type]['examples']) < 3:
                    by_type[issue_type]['examples'].append({
                        'test_id': result['test_id'],
                        'category': result['category'],
                        'description': issue['description'],
                        'impact': issue.get('impact', 'Unknown')
                    })

        return by_type

    def _analyze_by_category(self, analysis_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze quality issues by test category."""
        by_category = {}

        for result in analysis_results:
            category = result['category']
            if category not in by_category:
                by_category[category] = {
                    'total_tests': 0,
                    'tests_with_issues': 0,
                    'total_issues': 0,
                    'common_issue_types': {}
                }

            by_category[category]['total_tests'] += 1
            if result['issues']:
                by_category[category]['tests_with_issues'] += 1
                by_category[category]['total_issues'] += len(result['issues'])

                for issue in result['issues']:
                    issue_type = issue['type']
                    by_category[category]['common_issue_types'][issue_type] = \
                        by_category[category]['common_issue_types'].get(issue_type, 0) + 1

        # Calculate percentages
        for category, stats in by_category.items():
            stats['issue_rate'] = (stats['tests_with_issues'] / stats['total_tests'] * 100) \
                                  if stats['total_tests'] > 0 else 0

        return by_category

    def _severity_distribution(self, analysis_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Distribution of issues by severity level."""
        distribution = {'CRITICAL': [], 'HIGH': [], 'MEDIUM': [], 'LOW': []}

        for result in analysis_results:
            for issue in result['issues']:
                severity = issue.get('severity', 'MEDIUM')
                distribution[severity].append({
                    'test_id': result['test_id'],
                    'category': result['category'],
                    'type': issue['type'],
                    'description': issue['description']
                })

        return {
            'CRITICAL': {'count': len(distribution['CRITICAL']), 'issues': distribution['CRITICAL']},
            'HIGH': {'count': len(distribution['HIGH']), 'issues': distribution['HIGH']},
            'MEDIUM': {'count': len(distribution['MEDIUM']), 'issues': distribution['MEDIUM']},
            'LOW': {'count': len(distribution['LOW']), 'issues': distribution['LOW']}
        }

    def _extract_critical(self, analysis_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract all critical severity issues."""
        critical = []

        for result in analysis_results:
            for issue in result['issues']:
                if issue.get('severity') == 'CRITICAL':
                    critical.append({
                        'test_id': result['test_id'],
                        'category': result['category'],
                        'passed': result['test_result']['passed'],
                        'score': result['test_result']['validation_score'],
                        'issue_type': issue['type'],
                        'description': issue['description'],
                        'impact': issue.get('impact'),
                        'details': issue
                    })

        return critical

    def _generate_recommendations(self, analysis_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate actionable recommendations based on findings."""
        recommendations = []
        summary = self._generate_summary(analysis_results)

        # High false positive rate
        false_positive_rate = summary['false_positive_rate']
        if false_positive_rate > 10:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Validation Threshold',
                'issue': f'High false positive rate: {false_positive_rate:.1f}%',
                'recommendation': 'Consider raising pass threshold from 0.8 to 0.85 or improving validation criteria'
            })

        # Check for grounding issues
        grounding_issues = sum(1 for r in analysis_results
                               for i in r['issues']
                               if 'GROUNDING' in i['type'] or i['type'] == 'UNGROUNDED_FACTUAL_ANSWER')
        if grounding_issues > 0:
            recommendations.append({
                'priority': 'CRITICAL' if grounding_issues >= 3 else 'HIGH',
                'category': 'Grounding Enforcement',
                'issue': f'{grounding_issues} test(s) provide factual answers without grounding',
                'recommendation': 'Review RAG prompt to enforce grounding for factual answers. Ensure system refuses to answer without sources.'
            })

        # Check for impossible question handling
        impossible_menu_issues = sum(1 for r in analysis_results
                                     for i in r['issues']
                                     if i['type'] == 'IMPOSSIBLE_PROVIDES_MENU')
        if impossible_menu_issues > 0:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Impossible Questions',
                'issue': f'{impossible_menu_issues} impossible question(s) provide menus instead of "I don\'t know"',
                'recommendation': 'Update prompt to distinguish truly impossible vs ambiguous questions. Use direct uncertainty for unanswerable questions.'
            })

        # Check for completeness issues
        incomplete_issues = sum(1 for r in analysis_results
                               for i in r['issues']
                               if i['type'] in ['INCOMPLETE_CONTENT', 'PASSED_WITH_ISSUES'])
        if incomplete_issues > 3:
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'Content Completeness',
                'issue': f'{incomplete_issues} test(s) pass but have incomplete content',
                'recommendation': 'Review validation prompt to ensure completeness is properly weighted in scoring'
            })

        # Check for source count violations
        source_issues = sum(1 for r in analysis_results
                           for i in r['issues']
                           if i['type'] == 'INSUFFICIENT_SOURCES')
        if source_issues > 2:
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'Source Retrieval',
                'issue': f'{source_issues} test(s) have insufficient sources',
                'recommendation': 'Review RAG retrieval threshold and top_k settings to ensure sufficient context'
            })

        return recommendations

    def to_console(self, report: Dict[str, Any]) -> str:
        """Generate console-friendly summary."""
        lines = []
        lines.append("=" * 80)
        lines.append("QUALITY ANALYSIS SUMMARY")
        lines.append("=" * 80)

        summary = report['summary']
        lines.append(f"Total tests analyzed: {summary['total_tests']}")
        lines.append(f"Tests with quality issues: {summary['tests_with_quality_issues']}")
        lines.append(f"Total quality issues found: {summary['total_quality_issues']}")
        lines.append(f"False positives (passed with issues): {summary['false_positives']}")
        lines.append(f"False positive rate: {summary['false_positive_rate']:.1f}%")

        # Issue type distribution
        lines.append("\nIssue Type Distribution:")
        for issue_type, count in sorted(summary['issue_type_counts'].items(),
                                       key=lambda x: x[1], reverse=True):
            # Get severity distribution for this type
            breakdown = report['issue_breakdown'].get(issue_type, {})
            sev_dist = breakdown.get('severity_distribution', {})
            sev_summary = ", ".join(f"{k}: {v}" for k, v in sev_dist.items() if v > 0)
            lines.append(f"  {issue_type}: {count} ({sev_summary})")

        # Critical findings
        critical = report['critical_findings']
        if critical:
            lines.append(f"\nCRITICAL ISSUES ({len(critical)}):")
            for finding in critical[:5]:
                lines.append(f"  [{finding['test_id']}] {finding['issue_type']}: {finding['description']}")

        # Top false positives
        false_positives = report['false_positives']
        if false_positives:
            lines.append(f"\nTOP FALSE POSITIVES:")
            for fp in false_positives[:5]:
                lines.append(f"  [{fp['test_id']}] Score: {fp['score']:.2f}, Issues: {fp['issue_count']}")
                for issue in fp['issues'][:2]:
                    lines.append(f"    - [{issue['severity']}] {issue['type']}: {issue['description']}")

        # Recommendations
        recommendations = report.get('recommendations', [])
        if recommendations:
            lines.append("\nRECOMMENDATIONS:")
            for rec in recommendations:
                lines.append(f"  [{rec['priority']}] {rec['category']}: {rec['recommendation']}")

        lines.append("=" * 80)

        return "\n".join(lines)

    def to_markdown(self, report: Dict[str, Any]) -> str:
        """Generate markdown report."""
        lines = []
        lines.append("# Test Quality Analysis Report")
        lines.append(f"\n**Generated:** {report['timestamp']}")

        # Summary
        lines.append("\n## Summary\n")
        summary = report['summary']
        lines.append(f"- **Total tests analyzed:** {summary['total_tests']}")
        lines.append(f"- **Tests with quality issues:** {summary['tests_with_quality_issues']}")
        lines.append(f"- **Total quality issues:** {summary['total_quality_issues']}")
        lines.append(f"- **False positives:** {summary['false_positives']} ({summary['false_positive_rate']:.1f}%)")

        # Critical findings
        critical = report['critical_findings']
        if critical:
            lines.append(f"\n## Critical Issues ({len(critical)})\n")
            for finding in critical:
                lines.append(f"### {finding['test_id']}")
                lines.append(f"- **Category:** {finding['category']}")
                lines.append(f"- **Issue:** {finding['issue_type']}")
                lines.append(f"- **Description:** {finding['description']}")
                lines.append(f"- **Impact:** {finding['impact']}")
                lines.append(f"- **Passed:** {finding['passed']}, **Score:** {finding['score']:.2f}\n")

        # False positives
        false_positives = report['false_positives']
        if false_positives:
            lines.append(f"\n## False Positives ({len(false_positives)})\n")
            lines.append("Tests that passed but have quality issues:\n")
            for fp in false_positives:
                lines.append(f"### {fp['test_id']} (Score: {fp['score']:.2f})")
                lines.append(f"- **Category:** {fp['category']}")
                lines.append(f"- **Question:** {fp['question']}")
                lines.append(f"- **Issues ({fp['issue_count']}):**")
                for issue in fp['issues']:
                    lines.append(f"  - **[{issue['severity']}] {issue['type']}:** {issue['description']}")
                    if issue.get('impact'):
                        lines.append(f"    - *Impact:* {issue['impact']}")
                lines.append("")

        # Category analysis
        lines.append("\n## Analysis by Category\n")
        category_analysis = report['category_analysis']
        lines.append("| Category | Total Tests | Tests w/ Issues | Issue Rate |")
        lines.append("|----------|-------------|-----------------|------------|")
        for category, stats in sorted(category_analysis.items()):
            lines.append(f"| {category} | {stats['total_tests']} | {stats['tests_with_issues']} | {stats['issue_rate']:.1f}% |")

        # Recommendations
        recommendations = report.get('recommendations', [])
        if recommendations:
            lines.append("\n## Recommendations\n")
            for rec in recommendations:
                lines.append(f"### [{rec['priority']}] {rec['category']}")
                lines.append(f"**Issue:** {rec['issue']}")
                lines.append(f"\n**Recommendation:** {rec['recommendation']}\n")

        return "\n".join(lines)
