%
% superclass for all spectral projected gradient for l1-minimization (SPGL1) algorithms
%
% REQUIREMENTS:
%	- SPGL1 (https://github.com/mpf/spgl1 | https://friedlander.io/spgl1/)
%
% REFERENCES:
%	[1] E. van den Berg and M. P. Friedlander, "Probing the Pareto Frontier for Basis Pursuit Solutions",
%       SIAM J. Sci. Comput., Vol. 31, No. 2, pp. 890-912, DOI: 10.1137/080714488
%
% author: Martin F. Schiffner
% date: 2019-09-17
% modified: 2020-02-19
%
classdef spgl1 < regularization.algorithms.convex.convex

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        q ( 1, 1 ) double { mustBeMember( q, [ 1, 2 ] ) } = 1	% norm parameter

        % dependent properties
        spgl_opts ( 1, 1 ) { mustBeNonempty } = spgSetParms     % set default parameters

	end % properties

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = spgl1( rel_RMSEs, N_iterations_max, q )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure three arguments
            narginchk( 3, 3 );

            % ensure cell array for rel_RMSEs
            if ~iscell( rel_RMSEs )
                rel_RMSEs = { rel_RMSEs };
            end

            % superclass ensures valid rel_RMSEs
            % superclass ensures valid N_iterations_max
            % property validation functions ensure valid q

            % ensure equal number of dimensions and sizes
            [ rel_RMSEs, N_iterations_max, q ] = auxiliary.ensureEqualSize( rel_RMSEs, N_iterations_max, q );

            %--------------------------------------------------------------
            % 2.) create SPGL1 algorithms
            %--------------------------------------------------------------
            % constructor of superclass
            objects@regularization.algorithms.convex.convex( rel_RMSEs, N_iterations_max );

            % iterate SPGL1 algorithms
            for index_object = 1:numel( objects )

                %----------------------------------------------------------
                % a) set independent properties
                %----------------------------------------------------------
                objects( index_object ).q = q( index_object );

                %----------------------------------------------------------
                % b) set dependent properties
                %----------------------------------------------------------
                % create SPGL1 options structure
                objects( index_object ).spgl_opts = spgSetParms( 'verbosity', 1, 'optTol', 1e-4, 'iterations', objects( index_object ).N_iterations_max );

                % alternative projection methods for l2-minimization
                if objects( index_object ).q == 2
                    objects( index_object ).spgl_opts.project = @( x, weight, tau ) regularization.algorithms.convex.NormL2_project( x, weight, tau );
                    objects( index_object ).spgl_opts.primal_norm = @( x, weight ) regularization.algorithms.convex.NormL2_primal( x, weight );
                    objects( index_object ).spgl_opts.dual_norm = @( x, weight ) regularization.algorithms.convex.NormL2_dual( x, weight );
                end

            end % for index_object = 1:numel( objects )

        end % function objects = spgl1( rel_RMSEs, N_iterations_max, q )

        %------------------------------------------------------------------
        % string array (implement string method)
        %------------------------------------------------------------------
        function strs_out = string( algorithms_spgl1 )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class regularization.algorithms.convex.spgl1
            if ~isa( algorithms_spgl1, 'regularization.algorithms.convex.spgl1' )
                errorStruct.message = 'algorithms_spgl1 must be regularization.algorithms.convex.spgl1!';
                errorStruct.identifier = 'string:NoSPGL1RegularizationAlgorithms';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) create string array
            %--------------------------------------------------------------
            % initialize string array for strs_out
            strs_out = repmat( "", size( algorithms_spgl1 ) );

            % iterate SPGL1 algorithms
            for index_object = 1:numel( algorithms_spgl1 )

                strs_out( index_object ) = sprintf( "%s (q = %d)", 'SPGL1', algorithms_spgl1( index_object ).q );

            end % for index_object = 1:numel( algorithms_spgl1 )

        end % function strs_out = string( algorithms_spgl1 )

	end % methods

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (protected, hidden)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Access = protected, Hidden)

        %------------------------------------------------------------------
        % execute (implement execute_scalar method)
        %------------------------------------------------------------------
        function [ theta_recon, y_m_res, info ] = execute_scalar( algorithm_spgl1, op_A, y_m )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % superclass ensures class regularization.algorithms.algorithm (scalar) for algorithm_spgl1
            % superclass ensures numeric matrix or function_handle for op_A
            % superclass ensures compatibility of y_m with op_A

            %--------------------------------------------------------------
            % 2.) execute SPGL1
            %--------------------------------------------------------------
            % specify cell arrays
            theta_recon = cell( 1, numel( algorithm_spgl1.rel_RMSEs ) );
            y_m_res = cell( 1, numel( algorithm_spgl1.rel_RMSEs ) );
            info = cell( 1, numel( algorithm_spgl1.rel_RMSEs ) );

            % internal state for cold start
            theta_start = [];
            tau_start = [];

            % iterate rel_RMSEs
            for index_rel_RMSE = 1:numel( algorithm_spgl1.rel_RMSEs )

                % call SPGL1
                [ theta_recon{ index_rel_RMSE }, y_m_res{ index_rel_RMSE }, ~, info{ index_rel_RMSE } ] = spgl1( op_A, y_m, tau_start, algorithm_spgl1.rel_RMSEs( index_rel_RMSE ), theta_start, algorithm_spgl1.spgl_opts );

                % update internal state for warm start
                theta_start = theta_recon{ index_rel_RMSE };
                tau_start = info{ index_rel_RMSE }.tau;

            end % for index_rel_RMSE = 1:numel( algorithm_spgl1.rel_RMSEs )

            % concatenate horizontally
            theta_recon = cat( 2, theta_recon{ : } );
            y_m_res = cat( 2, y_m_res{ : } );
            info = cat( 2, info{ : } );

        end % function [ theta_recon, y_m_res, info ] = execute_scalar( algorithm_spgl1, op_A, y_m )

	end % methods (Access = protected, Hidden)

end % classdef spgl1 < regularization.algorithms.convex.convex
