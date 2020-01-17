%
% superclass for all spectral projected gradient for l1-minimization (SPGL1) options
%
% author: Martin F. Schiffner
% date: 2019-09-17
% modified: 2020-01-16
%
classdef algorithm_spgl1 < regularization.options.algorithm

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
            objects@regularization.options.algorithm( rel_RMSE, N_iterations_max );

            % iterate SPGL1 options
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).q = q( index_object );

            end % for index_object = 1:numel( objects )

        end % function objects = algorithm_spgl1( rel_RMSE, N_iterations_max, q )

        %------------------------------------------------------------------
        % string array (overload string method)
        %------------------------------------------------------------------
        function strs_out = string( algorithms_spgl1 )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class regularization.options.algorithm_spgl1
            if ~isa( algorithms_spgl1, 'regularization.options.algorithm_spgl1' )
                errorStruct.message = 'algorithms_spgl1 must be regularization.options.algorithm_spgl1!';
                errorStruct.identifier = 'string:NoOptionsSPGL1';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) create string array
            %--------------------------------------------------------------
            % initialize string array for strs_out
            strs_out = repmat( "", size( algorithms_spgl1 ) );

            % iterate SPGL1 options
            for index_object = 1:numel( algorithms_spgl1 )

                strs_out( index_object ) = sprintf( "%s (q = %d)", 'SPGL1', algorithms_spgl1( index_object ).q );

            end % for index_object = 1:numel( algorithms_spgl1 )

        end % function strs_out = string( algorithms_spgl1 )

	end % methods

end % classdef algorithm_spgl1 < regularization.options.algorithm