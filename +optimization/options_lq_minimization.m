%
% superclass for all lq-minimization options
%
% author: Martin F. Schiffner
% date: 2019-05-29
% modified: 2019-06-25
%
classdef options_lq_minimization < optimization.options

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        q ( 1, 1 ) double { mustBeNonnegative, mustBeLessThanOrEqual( q, 2 ) } = 1	% parameter for lq-quasinorm
%         tau ( :, 1 ) double = [] % parameter for SPGL1
        epsilon_n

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = options_lq_minimization( rel_RMSE, N_iterations_max, q, varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
%             % return if no input argument
%             if nargin == 0
%                 rel_RMSE
%                 N_iterations_max
%                 q = 
%             end

            % superclass ensures valid rel_RMSE
            % superclass ensures valid N_iterations_max

            % ensure correct value of the parameter q
            if ~all( ( q( : ) >= 0 & q( : ) <= 1 ) | q( : ) == 2 )
                errorStruct.message = 'The parameter q must satisfy 0 <= q <= 1 or q == 2!';
                errorStruct.identifier = 'lq_minimization:InvalidParameterQ';
                error( errorStruct );
            end

            % ensure tau
%             if nargin >= 5 && ~isempty( varargin{ 2 } )
%                 tau = varargin{ 2 };
%             else
%                 tau = repmat( [], size( rel_RMSE ) );
%             end

            % multiple rel_RMSE / single q
            if ~isscalar( rel_RMSE ) && isscalar( q )
                q = repmat( q, size( rel_RMSE ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( rel_RMSE, q );

            %--------------------------------------------------------------
            % 2.) create lq-minimization options
            %--------------------------------------------------------------
            % constructor of superclass
            objects@optimization.options( rel_RMSE, N_iterations_max, varargin{ : } );

            % iterate lq-minimization options
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).q = q( index_object );
%                 objects( index_object ).tau = tau( index_object );

            end % for index_object = 1:numel( objects )

        end % function objects = options_lq_minimization( rel_RMSE, N_iterations_max, q, varargin )

	end % methods

end % classdef options_lq_minimization < optimization.options
