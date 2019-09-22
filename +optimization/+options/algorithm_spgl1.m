%
% superclass for all spectral projected gradient for l1-minimization (SPGL1) options
%
% author: Martin F. Schiffner
% date: 2019-09-17
% modified: 2019-09-22
%
classdef algorithm_spgl1 < optimization.options.algorithm

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        q ( 1, 1 ) double { mustBeNonnegative, mustBeLessThanOrEqual( q, 2 ) } = 1	% norm parameter

	end % properties

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = algorithm_spgl1( rel_RMSE, N_iterations_max, q )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % superclass ensures valid rel_RMSE
            % superclass ensures valid N_iterations_max
            % property validation functions ensure valid q

            % multiple rel_RMSE / single q
            if ~isscalar( rel_RMSE ) && isscalar( q )
                q = repmat( q, size( rel_RMSE ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( rel_RMSE, q );

            %--------------------------------------------------------------
            % 2.) create SPGL1 options
            %--------------------------------------------------------------
            % constructor of superclass
            objects@optimization.options.algorithm( rel_RMSE, N_iterations_max );

            % iterate SPGL1 options
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).q = q( index_object );

            end % for index_object = 1:numel( objects )

        end % function objects = algorithm_spgl1( rel_RMSE, N_iterations_max, q )

	end % methods

end % classdef algorithm_spgl1 < optimization.options.algorithm
