%
% abstract superclass for all convex regularization algorithms
%
% author: Martin F. Schiffner
% date: 2020-02-15
% modified: 2020-02-16
%
classdef (Abstract) convex < regularization.algorithms.algorithm

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = convex( rel_RMSEs, N_iterations_max )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % superclass ensures valid rel_RMSEs
            % superclass ensures valid N_iterations_max

            %--------------------------------------------------------------
            % 2.) create convex regularization algorithms
            %--------------------------------------------------------------
            % constructor of superclass
            objects@regularization.algorithms.algorithm( rel_RMSEs, N_iterations_max )

        end % function objects = convex( rel_RMSEs, N_iterations_max )

	end % methods

end % classdef (Abstract) convex < regularization.algorithms.algorithm
