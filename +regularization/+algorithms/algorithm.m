%
% abstract superclass for all regularization algorithms
%
% author: Martin F. Schiffner
% date: 2019-09-17
% modified: 2021-02-11
%
classdef (Abstract) algorithm

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        rel_RMSEs ( :, 1 ) double { mustBeNonnegative, mustBeLessThanOrEqual( rel_RMSEs, 1 ) } = 0.3	% relative root-mean squared errors
        N_iterations_max ( 1, 1 ) double { mustBePositive, mustBeInteger, mustBeNonempty } = 1e3        % maximum number of iterations

	end % properties

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = algorithm( rel_RMSEs, N_iterations_max )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure two arguments
            narginchk( 2, 2 );

            % ensure cell array for rel_RMSEs
            if ~iscell( rel_RMSEs )
                rel_RMSEs = { rel_RMSEs };
            end

            % property validation functions ensure valid rel_RMSEs
            % property validation functions ensure valid N_iterations_max

            % ensure equal number of dimensions and sizes
            [ rel_RMSEs, N_iterations_max ] = auxiliary.ensureEqualSize( rel_RMSEs, N_iterations_max );

            %--------------------------------------------------------------
            % 2.) create regularization algorithms
            %--------------------------------------------------------------
            % repeat default regularization algorithm
            objects = repmat( objects, size( rel_RMSEs ) );

            % iterate regularization algorithms
            for index_object = 1:numel( objects )

                % ensure column vector for rel_RMSEs{ index_object }
                if ~iscolumn( rel_RMSEs{ index_object } )
                    errorStruct.message = sprintf( 'rel_RMSEs{ %d } must be a column vector!', index_object );
                    errorStruct.identifier = 'algorithm:NoColumnVector';
                    error( errorStruct );
                end

                % ensure strictly monotonic decreasing rel_RMSEs{ index_object }
                if ~issorted( rel_RMSEs{ index_object }, 'strictdescend' )
                    errorStruct.message = sprintf( 'rel_RMSEs{ %d } must be strictly monotonically decreasing!', index_object );
                    errorStruct.identifier = 'algorithm:NoStrictDecrease';
                    error( errorStruct );
                end

                % set independent properties
                objects( index_object ).rel_RMSEs = rel_RMSEs{ index_object };
                objects( index_object ).N_iterations_max = N_iterations_max( index_object );

            end % for index_object = 1:numel( objects )

        end % function objects = algorithm( rel_RMSEs, N_iterations_max )

        %------------------------------------------------------------------
        % execute algorithms
        %------------------------------------------------------------------
        function [ theta_recon, y_m_res, info ] = execute( algorithms, ops_A, y_m )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure three arguments
            narginchk( 3, 3 );

            % ensure class regularization.algorithms.algorithm
            if ~isa( algorithms, 'regularization.algorithms.algorithm' )
                errorStruct.message = 'algorithms must be regularization.algorithms.algorithm!';
                errorStruct.identifier = 'execute:NoRegularizationAlgorithms';
                error( errorStruct );
            end

            % ensure cell array for ops_A
            if ~iscell( ops_A )
                ops_A = { ops_A };
            end

            % ensure numeric matrices or function_handle for ops_A
            indicator = cellfun( @( x ) ~( isnumeric( x ) && ismatrix( x ) ) && ~isa( x, 'function_handle' ), ops_A );
            if any( indicator( : ) )
                errorStruct.message = 'ops_A must either be function_handle or numeric matrices!';
                errorStruct.identifier = 'execute:InvalidOperators';
                error( errorStruct );
            end

            % ensure cell array for y_m
            if ~iscell( y_m )
                y_m = { y_m };
            end

% TODO: ensure compatibility of y_m with ops_A

            % ensure equal number of dimensions and sizes
            [ algorithms, ops_A, y_m ] = auxiliary.ensureEqualSize( algorithms, ops_A, y_m );

            %--------------------------------------------------------------
            % 2.) execute algorithms
            %--------------------------------------------------------------
            % specify cell arrays
            theta_recon = cell( size( algorithms ) );
            y_m_res = cell( size( algorithms ) );
            info = cell( size( algorithms ) );

            % iterate algorithms
            for index_algorithm = 1:numel( algorithms )

                % call execute_scalar
                [ theta_recon{ index_algorithm }, y_m_res{ index_algorithm }, info{ index_algorithm } ] = execute_scalar( algorithms( index_algorithm ), ops_A{ index_algorithm }, y_m{ index_algorithm } );

            end % for index_algorithm = 1:numel( algorithms )

            % avoid cell arrays for single algorithms
            if isscalar( algorithms )
                theta_recon = theta_recon{ 1 };
                y_m_res = y_m_res{ 1 };
                info = info{ 1 };
            end

        end % function [ theta_recon, y_m_res, info ] = execute( algorithms, ops_A, y_m )

	end % methods

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (Abstract)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Abstract)

        %------------------------------------------------------------------
        % string array (overload string method)
        %------------------------------------------------------------------
        strs_out = string( algorithms )

	end % methods (Abstract)

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (Abstract, protected, hidden)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Abstract, Access = protected, Hidden)

        %------------------------------------------------------------------
        % execute regularization algorithm
        %------------------------------------------------------------------
        [ theta_recon, y_m_res, info ] = execute_scalar( algorithm, op_A, y_m )

	end % methods (Abstract, Access = protected, Hidden)

end % classdef (Abstract) algorithm
