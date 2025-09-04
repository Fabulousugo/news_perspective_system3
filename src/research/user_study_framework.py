# src/research/user_study_framework.py - User impact measurement

import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
from collections import defaultdict

from config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class UserSession:
    """User interaction session data"""
    session_id: str
    user_id: str
    timestamp: datetime
    articles_viewed: List[Dict]
    perspectives_explored: List[Dict]
    interaction_duration: float
    pre_study_responses: Optional[Dict] = None
    post_study_responses: Optional[Dict] = None

@dataclass
class PolarizationMeasurement:
    """Pre/post study polarization measurement"""
    user_id: str
    measurement_type: str  # 'pre' or 'post'
    timestamp: datetime
    political_orientation: int  # 1-7 scale (very liberal to very conservative)
    issue_positions: Dict[str, int]  # Issue -> position scale
    media_trust_scores: Dict[str, int]  # Source -> trust level
    attitude_certainty: Dict[str, int]  # Issue -> certainty level
    cross_cutting_exposure_willingness: int  # 1-5 scale

class UserStudyFramework:
    """Framework for conducting user impact studies"""
    
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or str(settings.DATA_DIR / "user_study.db")
        self._init_database()
        
    def _init_database(self):
        """Initialize SQLite database for user study data"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # User sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                articles_viewed TEXT,
                perspectives_explored TEXT,
                interaction_duration REAL,
                pre_study_responses TEXT,
                post_study_responses TEXT
            )
        ''')
        
        # Polarization measurements table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS polarization_measurements (
                measurement_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                measurement_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                political_orientation INTEGER,
                issue_positions TEXT,
                media_trust_scores TEXT,
                attitude_certainty TEXT,
                cross_cutting_exposure_willingness INTEGER
            )
        ''')
        
        # Article interactions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS article_interactions (
                interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                article_url TEXT NOT NULL,
                article_bias INTEGER,
                time_spent REAL,
                perspectives_viewed INTEGER,
                user_rating INTEGER,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES user_sessions (session_id)
            )
        ''')
        
        conn.commit()
        conn.close()
        
        logger.info(f"User study database initialized: {self.db_path}")
    
    def start_user_session(self, user_id: str) -> str:
        """Start a new user study session"""
        session_id = f"session_{user_id}_{int(datetime.now().timestamp())}"
        
        session = UserSession(
            session_id=session_id,
            user_id=user_id,
            timestamp=datetime.now(),
            articles_viewed=[],
            perspectives_explored=[],
            interaction_duration=0.0
        )
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO user_sessions 
            (session_id, user_id, timestamp, articles_viewed, perspectives_explored, interaction_duration)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            session.session_id,
            session.user_id,
            session.timestamp.isoformat(),
            json.dumps(session.articles_viewed),
            json.dumps(session.perspectives_explored),
            session.interaction_duration
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Started user session: {session_id}")
        return session_id
    
    def record_pre_study_measurement(self, user_id: str, responses: Dict) -> None:
        """Record pre-study polarization measurement"""
        measurement = PolarizationMeasurement(
            user_id=user_id,
            measurement_type='pre',
            timestamp=datetime.now(),
            political_orientation=responses.get('political_orientation', 4),
            issue_positions=responses.get('issue_positions', {}),
            media_trust_scores=responses.get('media_trust_scores', {}),
            attitude_certainty=responses.get('attitude_certainty', {}),
            cross_cutting_exposure_willingness=responses.get('cross_cutting_exposure_willingness', 3)
        )
        
        self._store_polarization_measurement(measurement)
    
    def record_post_study_measurement(self, user_id: str, responses: Dict) -> None:
        """Record post-study polarization measurement"""
        measurement = PolarizationMeasurement(
            user_id=user_id,
            measurement_type='post',
            timestamp=datetime.now(),
            political_orientation=responses.get('political_orientation', 4),
            issue_positions=responses.get('issue_positions', {}),
            media_trust_scores=responses.get('media_trust_scores', {}),
            attitude_certainty=responses.get('attitude_certainty', {}),
            cross_cutting_exposure_willingness=responses.get('cross_cutting_exposure_willingness', 3)
        )
        
        self._store_polarization_measurement(measurement)
    
    def _store_polarization_measurement(self, measurement: PolarizationMeasurement) -> None:
        """Store polarization measurement in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO polarization_measurements 
            (user_id, measurement_type, timestamp, political_orientation, 
             issue_positions, media_trust_scores, attitude_certainty, cross_cutting_exposure_willingness)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            measurement.user_id,
            measurement.measurement_type,
            measurement.timestamp.isoformat(),
            measurement.political_orientation,
            json.dumps(measurement.issue_positions),
            json.dumps(measurement.media_trust_scores),
            json.dumps(measurement.attitude_certainty),
            measurement.cross_cutting_exposure_willingness
        ))
        
        conn.commit()
        conn.close()
    
    def record_article_interaction(self, session_id: str, article_data: Dict, 
                                 time_spent: float, perspectives_viewed: int,
                                 user_rating: Optional[int] = None) -> None:
        """Record user interaction with an article"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO article_interactions 
            (session_id, article_url, article_bias, time_spent, perspectives_viewed, user_rating, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            session_id,
            article_data.get('url', ''),
            article_data.get('bias_label', 1),
            time_spent,
            perspectives_viewed,
            user_rating,
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def analyze_polarization_changes(self, user_id: Optional[str] = None) -> Dict:
        """Analyze polarization changes from pre to post measurements"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get pre and post measurements
        query = '''
            SELECT user_id, measurement_type, political_orientation, issue_positions,
                   media_trust_scores, attitude_certainty, cross_cutting_exposure_willingness
            FROM polarization_measurements
        '''
        
        params = []
        if user_id:
            query += ' WHERE user_id = ?'
            params.append(user_id)
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        conn.close()
        
        # Group by user
        user_measurements = defaultdict(lambda: {'pre': None, 'post': None})
        
        for row in results:
            uid, mtype, pol_orient, issue_pos, trust_scores, attitude_cert, cross_exposure = row
            
            user_measurements[uid][mtype] = {
                'political_orientation': pol_orient,
                'issue_positions': json.loads(issue_pos),
                'media_trust_scores': json.loads(trust_scores),
                'attitude_certainty': json.loads(attitude_cert),
                'cross_cutting_exposure_willingness': cross_exposure
            }
        
        # Calculate changes
        analysis = {
            'users_analyzed': 0,
            'political_orientation_changes': [],
            'media_trust_changes': [],
            'attitude_certainty_changes': [],
            'cross_cutting_willingness_changes': [],
            'summary': {}
        }
        
        for uid, measurements in user_measurements.items():
            if measurements['pre'] and measurements['post']:
                analysis['users_analyzed'] += 1
                
                # Political orientation change
                pol_change = measurements['post']['political_orientation'] - measurements['pre']['political_orientation']
                analysis['political_orientation_changes'].append(pol_change)
                
                # Cross-cutting exposure willingness
                cross_change = measurements['post']['cross_cutting_exposure_willingness'] - measurements['pre']['cross_cutting_exposure_willingness']
                analysis['cross_cutting_willingness_changes'].append(cross_change)
        
        # Calculate summary statistics
        if analysis['political_orientation_changes']:
            analysis['summary']['avg_political_orientation_change'] = np.mean(analysis['political_orientation_changes'])
            analysis['summary']['avg_cross_cutting_willingness_change'] = np.mean(analysis['cross_cutting_willingness_changes'])
            
            # Polarization reduction indicators
            analysis['summary']['users_with_reduced_extremism'] = len([
                change for change in analysis['political_orientation_changes']
                if abs(change) < 0  # Movement toward center
            ])
            
            analysis['summary']['users_with_increased_openness'] = len([
                change for change in analysis['cross_cutting_willingness_changes']
                if change > 0  # Increased willingness to see other perspectives
            ])
        
        return analysis
    
    def get_interaction_patterns(self, session_id: Optional[str] = None) -> Dict:
        """Analyze user interaction patterns"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = '''
            SELECT session_id, article_bias, time_spent, perspectives_viewed, user_rating
            FROM article_interactions
        '''
        
        params = []
        if session_id:
            query += ' WHERE session_id = ?'
            params.append(session_id)
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        conn.close()
        
        # Analyze patterns
        bias_interactions = defaultdict(list)
        total_time_by_bias = defaultdict(float)
        perspective_exploration = []
        
        for row in results:
            sid, bias, time_spent, perspectives_viewed, rating = row
            
            bias_name = {0: 'left', 1: 'center', 2: 'right'}.get(bias, 'unknown')
            bias_interactions[bias_name].append({
                'time_spent': time_spent,
                'perspectives_viewed': perspectives_viewed,
                'rating': rating
            })
            
            total_time_by_bias[bias_name] += time_spent
            perspective_exploration.append(perspectives_viewed)
        
        analysis = {
            'total_interactions': len(results),
            'time_spent_by_bias': dict(total_time_by_bias),
            'avg_perspectives_per_article': np.mean(perspective_exploration) if perspective_exploration else 0,
            'cross_cutting_exposure_rate': len([p for p in perspective_exploration if p > 0]) / len(perspective_exploration) if perspective_exploration else 0,
            'bias_distribution': {bias: len(interactions) for bias, interactions in bias_interactions.items()}
        }
        
        return analysis

# Pre-built survey questions for user studies
POLARIZATION_SURVEY_QUESTIONS = {
    'pre_study': {
        'political_orientation': {
            'question': 'On political matters, how would you describe your views?',
            'scale': '1 (Very Liberal) to 7 (Very Conservative)',
            'type': 'likert_7'
        },
        'issue_positions': {
            'climate_change': 'Climate change requires immediate government action (1=Strongly Disagree, 5=Strongly Agree)',
            'healthcare': 'Government should provide universal healthcare (1=Strongly Disagree, 5=Strongly Agree)',
            'immigration': 'Immigration levels should be increased (1=Strongly Disagree, 5=Strongly Agree)',
            'gun_control': 'Gun control laws should be stricter (1=Strongly Disagree, 5=Strongly Agree)',
            'taxes': 'Taxes on wealthy should be increased (1=Strongly Disagree, 5=Strongly Agree)'
        },
        'media_trust': {
            'question': 'How much do you trust the following news sources?',
            'sources': ['CNN', 'Fox News', 'Reuters', 'NPR', 'BBC', 'Wall Street Journal'],
            'scale': '1 (Not at all) to 5 (Completely)'
        },
        'cross_cutting_exposure_willingness': {
            'question': 'How willing are you to read news from sources that typically disagree with your political views?',
            'scale': '1 (Very Unwilling) to 5 (Very Willing)',
            'type': 'likert_5'
        }
    },
    'post_study': {
        # Same questions as pre-study, plus:
        'system_effectiveness': {
            'question': 'How helpful was seeing alternative perspectives on news stories?',
            'scale': '1 (Not helpful) to 5 (Very helpful)',
            'type': 'likert_5'
        },
        'perspective_quality': {
            'question': 'How accurate were the alternative perspectives shown?',
            'scale': '1 (Very inaccurate) to 5 (Very accurate)',
            'type': 'likert_5'
        },
        'future_usage': {
            'question': 'How likely are you to seek out diverse perspectives on news in the future?',
            'scale': '1 (Very unlikely) to 5 (Very likely)',
            'type': 'likert_5'
        }
    }
}

# Example usage
if __name__ == "__main__":
    # Initialize user study framework
    study = UserStudyFramework()
    
    # Example pre-study measurement
    pre_responses = {
        'political_orientation': 3,  # Slightly liberal
        'issue_positions': {
            'climate_change': 4,
            'healthcare': 3,
            'immigration': 2,
            'gun_control': 4,
            'taxes': 3
        },
        'media_trust_scores': {
            'CNN': 3,
            'Fox News': 2,
            'Reuters': 4,
            'NPR': 4
        },
        'cross_cutting_exposure_willingness': 2
    }
    
    study.record_pre_study_measurement('user_001', pre_responses)
    
    # Start session
    session_id = study.start_user_session('user_001')
    
    # Record article interaction
    article_data = {
        'url': 'https://example.com/article1',
        'bias_label': 2,  # Right-leaning
        'title': 'Sample Article'
    }
    
    study.record_article_interaction(
        session_id=session_id,
        article_data=article_data,
        time_spent=120.5,
        perspectives_viewed=2,
        user_rating=4
    )
    
    print("User study framework demo completed")