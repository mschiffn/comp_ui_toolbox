%
% superclass for all lq-minimization options
%
% author: Martin F. Schiffner
% date: 2019-05-29
% modified: 2020-02-26
%
classdef lq_minimization < regularization.options.common

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        algorithm ( 1, 1 ) regularization.algorithms.algorithm { mustBeNonempty } = regularization.algorithms.convex.spgl1( 0.3, 1e3, 1 )	% algorithm options
        save_progress ( 1, 1 ) logical { mustBeNonempty } = true	% save intermediate results

	end % properties

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = lq_minimization( options_common, varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure nonempty options_common
            if nargin < 1 || isempty( options_common )
                options_common = regularization.options.common;
            end

            % ensure class regularization.options.common
            if ~isa( options_common, 'regularization.options.common' )
                errorStruct.message = 'options_common must be regularization.options.common!';
                errorStruct.identifier = 'lq_minimization:NoCommonOptions';
                error( errorStruct );
            end

            % ensure equal number of dimensions and sizes
            [ options_common, varargin{ : } ] = auxiliary.ensureEqualSize( options_common, varargin{ : } );

            %--------------------------------------------------------------
            % 2.) create lq-minimization options
            %--------------------------------------------------------------
            % constructor of superclass
            objects@regularization.options.common( regularization.options.energy_rx( [ options_common.momentary ], [ options_common.tgc ], [ options_common.dictionary ] ), [ options_common.normalization ] );

            % reshape lq-minimization options
            objects = reshape( objects, size( options_common ) );

            % iterate lq-minimization options
            for index_object = 1:numel( objects )

                % iterate arguments
                for index_arg = 1:numel( varargin )

                    if isa( varargin{ index_arg }, 'regularization.algorithms.algorithm' )

                        %--------------------------------------------------
                        % a) algorithm options
                        %--------------------------------------------------
                        objects( index_object ).algorithm = varargin{ index_arg }( index_object );

                    else

                        %--------------------------------------------------
                        % b) unknown class
                        %--------------------------------------------------
                        errorStruct.message = sprintf( 'Class of varargin{ %d } is unknown!', index_arg );
                        errorStruct.identifier = 'lq_minimization:UnknownClass';
                        error( errorStruct );

                    end % if isa( varargin{ index_arg }, 'regularization.algorithms.algorithm' )

                end % for index_arg = 1:numel( varargin )

            end % for index_object = 1:numel( objects )

        end % function objects = lq_minimization( options_common, varargin )

        %------------------------------------------------------------------
        % display options
        %------------------------------------------------------------------
        function show( options )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class regularization.options.lq_minimization
            if ~isa( options, 'regularization.options.lq_minimization' )
                errorStruct.message = 'options must be regularization.options.lq_minimization!';
                errorStruct.identifier = 'show:NoOptions';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) display options
            %--------------------------------------------------------------
            % iterate lq-minimization options
            for index_object = 1:numel( options )

                % print header
                auxiliary.print_header( "lq-minimization options", '=' );

                % print properties
                fprintf( ' %-12s: %-13s\n', 'algorithm', string( options( index_object ).algorithm ) );

                % call method show of superclass
                show@regularization.options.common( options( index_object ) );

            end % for index_object = 1:numel( options )

        end % function show( options )

	end % methods

end % classdef lq_minimization < regularization.options.common
