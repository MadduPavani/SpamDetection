a
    e��b�  �                   @   s(   d Z dd� Zdd� Zdd� Zdd� Zd	S )
z8Common utilities for registering LinearOperator methods.c                 C   s@   | j r|j rdS | j du r$|j du s8| j du r<|j du r<dS dS )z(Get combined hint for self-adjoint-ness.TFN)�is_self_adjoint��
operator_a�
operator_b� r   �zC:\Users\cnaga\AppData\Local\Programs\Python\Python39\Lib\site-packages\tensorflow/python/ops/linalg/registrations_util.py�$combined_commuting_self_adjoint_hint   s    
���r   c                 C   sp   | j r|j rdS | j du rH|j du rH| j}|j}|durH|durH||kS | j |j krl| j durl|j durldS dS )z3Return a hint to whether the composition is square.TFN)�	is_squareZrange_dimensionZdomain_dimension)r   r   �m�lr   r   r   r   +   s    ��r   c                 C   s0   | j du r,| jdu r,|j du r,|jdu r,dS dS )z&Get combined PD hint for compositions.TN)Zis_positive_definiter   r   r   r   r   �)combined_commuting_positive_definite_hint@   s    
���r   c                 C   s$   | j du s|j du rdS | j o"|j S )zGet combined hint for when .F)Zis_non_singularr   r   r   r   �combined_non_singular_hintM   s
    
�r   N)�__doc__r   r   r   r   r   r   r   r   �<module>   s                                                                                                                                                                                                                                                             