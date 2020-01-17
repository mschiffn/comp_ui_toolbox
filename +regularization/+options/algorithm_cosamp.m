%
% superclass for all compressive sampling matching pursuit (CoSaMP) options
%
% author: Martin F. Schiffner
% date: 2019-09-22
% modified: 2020-01-16
%
classdef algorithm_cosamp < regularization.options.algorithm

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        sparsity ( 1, 1 ) double { mustBePositive, mustBeInteger } = 10	% sparsity level

	end % properties

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = algorithm_cosamp( rel_RMSE, N_iterations_max, sparsity )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % superclass ensures valid rel_RMSE
            % superclass ensures valid N_iterations_max
            % property validation functions ensure valid sparsity

            % multiple rel_RMSE / single q
            if ~isscalar( rel_RMSE ) && isscalar( sparsity )
                sparsity = repmat( sparsity, size( rel_RMSE ) );
            end

            %--------------------------------------------------------------
            % 2.) create CoSaMP options
            %--------------------------------------------------------------
            % constructor of superclass
            objects@regularization.options.algorithm( rel_RMSE, N_iterations_max );

            % iterate CoSaMP options
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).sparsity = sparsity( index_object );

            end % for index_object = 1:numel( objects )

        end % function objects = algorithm_cosamp( rel_RMSE, N_iterations_max )

        %------------------------------------------------------------------
        % string array (overload string method)
        %------------------------------------------------------------------
        function strs_out = string( algorithms_cosamp )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class regularization.options.algorithm_cosamp
            if ~isa( algorithms_cosamp, 'regularization.options.algorithm_cosamp' )
                errorStruct.message = 'algorithms_cosamp must be regularization.options.algorithm_cosamp!';
                errorStruct.identifier = 'string:NoOptionsCoSaMP';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) create string array
            %--------------------------------------------------------------
            % initialize string array for strs_out
            strs_out = repmat( "", size( algorithms_cosamp ) );

            % iterate SPGL1 options
            for index_object = 1:numel( algorithms_cosamp )

                strs_out( index_object ) = sprintf( "%s (s = %d)", 'CoSaMP', algorithms_cosamp( index_object ).sparsity );

            end % for index_object = 1:numel( algorithms_cosamp )

        end % function strs_out = string( algorithms_cosamp )

	end % methods

end % classdef algorithm_cosamp < regularization.options.algorithm
