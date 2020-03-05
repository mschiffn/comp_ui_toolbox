%
% superclass for all spatial anti-aliasing options
%
% author: Martin F. Schiffner
% date: 2020-03-03
% modified: 2020-03-04
%
classdef anti_aliasing

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        tx ( 1, 1 ) scattering.anti_aliasing_filters.anti_aliasing_filter { mustBeNonempty } = scattering.anti_aliasing_filters.raised_cosine( 0.5 )	% spatial anti-aliasing filter (tx)
        rx ( 1, 1 ) scattering.anti_aliasing_filters.anti_aliasing_filter { mustBeNonempty } = scattering.anti_aliasing_filters.raised_cosine( 0.5 )	% spatial anti-aliasing filter (rx)

	end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = anti_aliasing( tx, rx )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure nonempty tx
            if nargin < 1 || isempty( tx )
                tx = scattering.anti_aliasing_filters.raised_cosine( 0.5 );
            end

            % ensure nonempty rx
            if nargin < 2 || isempty( rx )
                rx = scattering.anti_aliasing_filters.raised_cosine( 0.5 );
            end

            % multiple tx / single rx
            if ~isscalar( tx ) && isscalar( rx )
                rx = repmat( rx, size( tx ) );
            end

            % single tx / multiple rx
            if isscalar( tx ) && ~isscalar( rx )
                tx = repmat( tx, size( rx ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( tx, rx );

            %--------------------------------------------------------------
            % 2.) create spatial anti-aliasing options
            %--------------------------------------------------------------
            % repeat default spatial anti-aliasing options
            objects = repmat( objects, size( tx ) );

            % iterate spatial anti-aliasing options
            for index_object = 1:numel( tx )

                % set independent properties
                objects( index_object ).tx = tx( index_object );
                objects( index_object ).rx = rx( index_object );

            end % for index_object = 1:numel( tx )

        end % function objects = anti_aliasing( tx, rx )

        %------------------------------------------------------------------
        % string array (implement string method)
        %------------------------------------------------------------------
        

	end % methods

end % classdef anti_aliasing
