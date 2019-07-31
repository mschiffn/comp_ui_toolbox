%
% superclass for all momentary scattering operator options
%
% author: Martin F. Schiffner
% date: 2019-07-09
% modified: 2019-07-30
%
classdef options_momentary

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        sequence ( 1, 1 ) scattering.options_sequence { mustBeNonempty } = scattering.options_sequence_full                         % sequence options
        anti_aliasing ( 1, 1 ) scattering.options_anti_aliasing { mustBeNonempty } = scattering.options_anti_aliasing_cosine( 0.1 )	% spatial anti-aliasing filter options
        gpu ( 1, 1 ) scattering.options_gpu { mustBeNonempty } = scattering.options_gpu_active( 0 )                                 % GPU options
        algorithm ( 1, 1 ) scattering.options_algorithm { mustBeNonempty } = scattering.options_algorithm_direct                    % algorithm options

    end % properties

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function object = options_momentary( varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % check number of arguments
            if nargin == 0
                return;
            end

            %--------------------------------------------------------------
            % 2.) create momentary scattering operator options
            %--------------------------------------------------------------
            object = set_properties( object, varargin{ : } );

        end % function object = options_momentary( varargin )

        %------------------------------------------------------------------
        % set properties
        %------------------------------------------------------------------
        function options_momentary = set_properties( options_momentary, varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.options_momentary (scalar)
            if ~( isa( options_momentary, 'scattering.options_momentary' ) && isscalar( options_momentary ) )
                errorStruct.message = 'options_momentary must be scattering.options_momentary!';
                errorStruct.identifier = 'set_properties:NoOptionsMomentary';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) set properties
            %--------------------------------------------------------------
            % iterate arguments
            for index_arg = 1:numel( varargin )

                if isa( varargin{ index_arg }, 'scattering.options_sequence' )

                    %------------------------------------------------------
                    % a) sequence options
                    %------------------------------------------------------
                    options_momentary.sequence = varargin{ index_arg };

                elseif isa( varargin{ index_arg }, 'scattering.options_anti_aliasing' )

                    %------------------------------------------------------
                    % b) spatial anti-aliasing filter options
                    %------------------------------------------------------
                    options_momentary.anti_aliasing = varargin{ index_arg };

                elseif isa( varargin{ index_arg }, 'scattering.options_gpu' )

                    %------------------------------------------------------
                    % c) GPU options
                    %------------------------------------------------------
                    options_momentary.gpu = varargin{ index_arg };

                elseif isa( varargin{ index_arg }, 'scattering.options_algorithm' )

                    %------------------------------------------------------
                    % d) algorithm options
                    %------------------------------------------------------
                    options_momentary.algorithm = varargin{ index_arg };

                else

                    %------------------------------------------------------
                    % e) unknown class
                    %------------------------------------------------------
                    errorStruct.message = sprintf( 'Class of varargin{ %d } is unknown!', index_arg );
                    errorStruct.identifier = 'set_properties:UnknownClass';
                    error( errorStruct );

                end

            end % for index_arg = 1:numel( varargin )

        end % function options_momentary = set_properties( options_momentary, varargin )

	end % methods

end % classdef options_momentary
