%
% superclass for all reweighted regularization algorithms
%
% REFERENCES:
%	[1] S. Foucart and M. J. Lai, "Sparsest solutions of underdetermined linear systems via lq-minimization for 0 < q <= 1",
%       Appl. Comput. Harmon. Anal., Vol. 26, No. 3, Dec. 2009, pp. 395-407, DOI: 10.1016/j.acha.2008.09.001
%
% author: Martin F. Schiffner
% date: 2020-02-13
% modified: 2020-02-16
%
classdef reweighting < regularization.algorithms.algorithm

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        algorithm ( 1, 1 ) regularization.algorithms.algorithm { mustBeNonempty } = regularization.algorithms.spgl1( 0.3, 1e3 )	% regularization algorithm
        q ( 1, 1 ) double { mustBeNonnegative, mustBeLessThanOrEqual( q, 2 ) } = 0.5	% norm parameter
        epsilon_n ( :, 1 ) double { mustBeNonnegative } = 1 ./ ( 1 + (1:5) )            % monotonically decreasing reweighting sequence

        % dependent properties
        N_iterations ( 1, 1 ) double { mustBePositive, mustBeInteger, mustBeNonempty } = 5  % number of reweighting iterations

	end % properties

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = reweighting( algorithms, q, epsilon_n )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class regularization.algorithms.algorithm
            if ~isa( algorithms, 'regularization.algorithms.algorithm' )
                errorStruct.message = 'algorithms must be regularization.algorithms.algorithm!';
                errorStruct.identifier = 'reweighting:NoRegularizationAlgorithms';
                error( errorStruct );
            end

            % property validation function ensures nonnegative doubles for q

            % ensure cell array for epsilon_n
            if ~iscell( epsilon_n )
                epsilon_n = { epsilon_n };
            end

            % property validation function ensures nonnegative doubles for epsilon_n

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( algorithms, q, epsilon_n );

            %--------------------------------------------------------------
            % 2.) create reweighted regularization algorithms
            %--------------------------------------------------------------
            % extract relative RMSEs
            rel_RMSEs = reshape( { algorithms.rel_RMSEs }, size( algorithms ) );
            N_iterations_max = reshape( [ algorithms.N_iterations_max ], size( algorithms ) );

            % constructor of superclass
            objects@regularization.algorithms.algorithm( rel_RMSEs, N_iterations_max );

            % iterate reweighted regularization algorithms
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).algorithm = algorithms( index_object );
                objects( index_object ).q = q( index_object );
                objects( index_object ).epsilon_n = epsilon_n{ index_object };

                % set dependent properties
                objects( index_object ).N_iterations = numel( objects( index_object ).epsilon_n );

            end % for index_object = 1:numel( objects )

        end % function objects = reweighting( algorithms, q, epsilon_n )

        %------------------------------------------------------------------
        % string array (implement string method)
        %------------------------------------------------------------------
        function strs_out = string( algorithms_reweighting )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class regularization.algorithms.reweighting
            if ~isa( algorithms_reweighting, 'regularization.algorithms.reweighting' )
                errorStruct.message = 'algorithms_reweighting must be regularization.algorithms.reweighting!';
                errorStruct.identifier = 'string:NoReweightingRegularizationAlgorithms';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) create string array
            %--------------------------------------------------------------
            % initialize strings
            strs_out = repmat( "", size( algorithms_reweighting ) );

            % iterate reweighted regularization algorithms
            for index_object = 1:numel( algorithms_reweighting )

                strs_out( index_object ) = sprintf( "reweighting (q = %.2f, epsilon = %s)", algorithms_reweighting( index_object ).q, strjoin( string( algorithms_reweighting( index_object ).epsilon_n ), '|' ) );

            end % for index_object = 1:numel( algorithms_reweighting )

        end % function strs_out = string( algorithms_reweighting )

	end % methods

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (protected, hidden)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Access = protected, Hidden)

        %------------------------------------------------------------------
        % execute (implement execute_scalar method)
        %------------------------------------------------------------------
        function [ theta_recon_n, y_m_res_n, info ] = execute_scalar( algorithms_reweighting, op_A, y_m )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % superclass ensures class regularization.algorithms.algorithm (scalar) for spgl1
            % superclass ensures numeric matrix or function_handle for op_A
            % superclass ensures compatibility of y_m

            %--------------------------------------------------------------
            % 2.) execute reweighting
            %--------------------------------------------------------------
% TODO: warm start for multiple rel_RMSEs
            % compute initial guess
            [ theta_recon_n, y_m_res_n, info ] = execute_scalar( algorithms_reweighting.algorithm, op_A, y_m );

            % allocate memory for results and specify start vector ( minimizer of P_{(1, eta)} )
            theta_recon_n = [ theta_recon_n, zeros( size( theta_recon_n, 1 ), algorithms_reweighting.N_iterations ) ];
            y_m_res_n = [ y_m_res_n, zeros( size( y_m_res_n, 1 ), algorithms_reweighting.N_iterations ) ];

            % statistics
            info.info_reweighting = cell( 1, algorithms_reweighting.N_iterations );

            % iterate reweighted problems
            for index_iter = 1:algorithms_reweighting.N_iterations
% TODO: residual voltages

                % specify diagonal weighting matrix
                weights_act = ( abs( theta_recon_n( :, index_iter ) ) + algorithms_reweighting.epsilon_n( index_iter ) ).^( 1 - algorithms_reweighting.q );
                LT_act_n = linear_transforms.composition( linear_transforms.weighting( weights_act ), LT_act );

                % define anonymous function for reweighted sensing matrix
% TODO: use function handle...
                op_A_bar_n = @( x, mode ) combined_quick( operator_born_act, mode, x, LT_act_n, LT_tgc_act );

                % solve reweighted problem
                [ temp, temp_res, info.info_reweighting{ index_iter } ] = execute_scalar( algorithms_reweighting.algorithm, op_A_bar_n, y_m );

                % remove weights
                theta_recon_n( :, index_iter + 1 ) = temp .* weights_act;
                y_m_res_n( :, index_iter + 1 ) = temp_res;

            end % for index_iter = 1:algorithms_reweighting.N_iterations

        end % function [ theta_recon_n, y_m_res_n, info ] = execute_scalar( algorithms_reweighting, op_A, y_m )

	end % methods (Access = protected, Hidden)

end % classdef reweighting < regularization.algorithms.algorithm
