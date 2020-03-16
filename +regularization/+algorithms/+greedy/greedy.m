%
% abstract superclass for all greedy regularization algorithms
%
% author: Martin F. Schiffner
% date: 2020-02-15
% modified: 2020-02-16
%
classdef (Abstract) greedy < regularization.algorithms.algorithm

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = greedy( rel_RMSEs, N_iterations_max )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % superclass ensures valid rel_RMSEs
            % superclass ensures valid N_iterations_max

            %--------------------------------------------------------------
            % 2.) create greedy regularization algorithms
            %--------------------------------------------------------------
            % constructor of superclass
            objects@regularization.algorithms.algorithm( rel_RMSEs, N_iterations_max );

        end % function objects = greedy( rel_RMSEs, N_iterations_max )

	end % methods

end % classdef (Abstract) greedy < regularization.algorithms.algorithm
