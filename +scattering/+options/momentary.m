%
% superclass for all momentary scattering operator options
%
% author: Martin F. Schiffner
% date: 2019-07-09
% modified: 2019-08-03
%
classdef momentary

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        sequence ( 1, 1 ) scattering.options.sequence { mustBeNonempty } = scattering.options.sequence_full                         % sequence options
        anti_aliasing ( 1, 1 ) scattering.options.anti_aliasing { mustBeNonempty } = scattering.options.anti_aliasing_raised_cosine( 0.5 )	% spatial anti-aliasing filter options
        gpu ( 1, 1 ) scattering.options.gpu { mustBeNonempty } = scattering.options.gpu_active( 0 )                                 % GPU options
        algorithm ( 1, 1 ) scattering.options.algorithm { mustBeNonempty } = scattering.options.algorithm_direct                    % algorithm options

    end % properties

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function object = momentary( varargin )

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

        end % function object = momentary( varargin )

        %------------------------------------------------------------------
        % set properties
        %------------------------------------------------------------------
        function momentary = set_properties( momentary, varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.options.momentary (scalar)
            if ~( isa( momentary, 'scattering.options.momentary' ) && isscalar( momentary ) )
                errorStruct.message = 'momentary must be scattering.options.momentary!';
                errorStruct.identifier = 'set_properties:NoOptionsMomentary';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) set properties
            %--------------------------------------------------------------
            % iterate arguments
            for index_arg = 1:numel( varargin )

                if isa( varargin{ index_arg }, 'scattering.options.sequence' )

                    %------------------------------------------------------
                    % a) sequence options
                    %------------------------------------------------------
                    momentary.sequence = varargin{ index_arg };

                elseif isa( varargin{ index_arg }, 'scattering.options.anti_aliasing' )

                    %------------------------------------------------------
                    % b) spatial anti-aliasing filter options
                    %------------------------------------------------------
                    momentary.anti_aliasing = varargin{ index_arg };

                elseif isa( varargin{ index_arg }, 'scattering.options.gpu' )

                    %------------------------------------------------------
                    % c) GPU options
                    %------------------------------------------------------
                    momentary.gpu = varargin{ index_arg };

                elseif isa( varargin{ index_arg }, 'scattering.options.algorithm' )

                    %------------------------------------------------------
                    % d) algorithm options
                    %------------------------------------------------------
                    momentary.algorithm = varargin{ index_arg };

                else

                    %------------------------------------------------------
                    % e) unknown class
                    %------------------------------------------------------
                    errorStruct.message = sprintf( 'Class of varargin{ %d } is unknown!', index_arg );
                    errorStruct.identifier = 'set_properties:UnknownClass';
                    error( errorStruct );

                end

            end % for index_arg = 1:numel( varargin )

        end % function momentary = set_properties( momentary, varargin )

        %------------------------------------------------------------------
        % show
        %------------------------------------------------------------------
				

	end % methods

end % classdef momentary
